import * as d3 from 'd3';

export function activationMemory(
    a, // attention heads
    b, // micro batch size
    h, // hidden dimension size
    h_ff, // feedforward dimension size (often h_ff = 4h)
    L, // number of layers
    s, // sequence length
    v, // vocab size
    tp = 1, // tensor model parallelism
    mixed = true,
    recomputation = "none",
    ff_activation = "relu",
    seq_parallel = false
) {
    console.log('activationMemory called with:', { a, b, h, h_ff, L, s, v, tp, mixed, recomputation, ff_activation, seq_parallel });
    // https://arxiv.org/pdf/2205.05198
    const bytesPerValue = mixed ? 2 : 4;

    let oneLayerAttention;
    if (recomputation === "none" || recomputation === "full") {
        if (seq_parallel) {
            oneLayerAttention = s * b * h / tp * (bytesPerValue * 5 + 1) + ((2 * bytesPerValue + 1) * a * s * s * b); // eq (2)
        } else {
            oneLayerAttention = s * b * h * (bytesPerValue * 4 / tp + bytesPerValue + 1) + ((2 * bytesPerValue + 1) * a * s * s * b / tp); // eq (2)
        }
    } else if (recomputation === "selective") {
        if (seq_parallel) {
            oneLayerAttention = s * b * h / tp * (bytesPerValue * 5 + 1); // table 2
        } else {
            oneLayerAttention = s * b * h * (bytesPerValue * 4 / tp + bytesPerValue + 1); // table 2
        }
    } else {
        throw new Error("Invalid recomputation value");
    }

    let oneLayerFeedforward;
    if (ff_activation === "relu") {
        if (seq_parallel) {
            oneLayerFeedforward = (s * b * h * bytesPerValue / tp + (s * b * h_ff * bytesPerValue / tp) // inputs of 1st/2nd linear layers
                + s * b * h / tp);  // dropout
        } else {
            oneLayerFeedforward = (s * b * h * bytesPerValue + (s * b * h_ff * bytesPerValue / tp) // inputs of 1st/2nd linear layers
                + s * b * h);  // dropout
        }
    } else if (ff_activation === "gelu") {
        if (seq_parallel) {
            oneLayerFeedforward = (s * b * h * bytesPerValue / tp + (s * b * h_ff * bytesPerValue / tp) // inputs of 1st/2nd linear layers
                + s * b * h_ff * bytesPerValue / tp // inputs of activation function (not really necessary for Relu)
                + s * b * h / tp);  // dropout
        } else {
            oneLayerFeedforward = (s * b * h * bytesPerValue + (s * b * h_ff * bytesPerValue / tp) // inputs of 1st/2nd linear layers
                + s * b * h_ff * bytesPerValue / tp // inputs of activation function (not really necessary for Relu)
                + s * b * h);  // dropout
        }
    } else if (ff_activation === "swiglu") {
        if (seq_parallel) {
            oneLayerFeedforward = (s * b * h * bytesPerValue / tp + (s * b * h_ff * bytesPerValue / tp) // inputs of input/output linear layers
                + s * b * h_ff * bytesPerValue * 3 / tp // inputs of activation function
                + s * b * h / tp);  // dropout (note that dropout is lower-precision - boolean)
        } else {
            oneLayerFeedforward = (s * b * h * bytesPerValue + (s * b * h_ff * bytesPerValue / tp) // inputs of input/output linear layers
                + s * b * h_ff * bytesPerValue * 3 / tp // inputs of activation function
                + s * b * h);  // dropout (note that dropout is lower-precision - boolean)
        }
    }

    let layerNorm;
    if (seq_parallel) {
        layerNorm = s * b * h * bytesPerValue / tp;
    } else {
        layerNorm = s * b * h * bytesPerValue;
    }

    const inputDropout = seq_parallel ? s * b * h / tp : s * b * h; // section 4.3
    const outputLayerNorm = seq_parallel ? s * b * h * bytesPerValue / tp : s * b * h * bytesPerValue;
    const outputLayerProjection = seq_parallel ? s * b * h * bytesPerValue / tp : s * b * h * bytesPerValue;
    const outputCrossEntropy = seq_parallel ? s * b * v * 4 / tp : s * b * v * 4;  // In FP32


    let data
    if (recomputation === "none" || recomputation === "selective") {

        data = {
            name: "Activation Memory",
            children: [
                ...Array.from({ length: L }, (_, index) => ({
                    name: `Layer ${index + 1}`,
                    children: [
                        { name: 'Attention', value: oneLayerAttention },
                        { name: 'Feedforward', value: oneLayerFeedforward },
                        { name: 'LayerNorm', value: 2 * layerNorm },
                    ]
                })),
                { name: 'Dropout', value: inputDropout },
                { name: 'LayerNorm', value: outputLayerNorm },
                { name: 'Projection', value: outputLayerProjection },
                { name: 'Cross Entropy', value: outputCrossEntropy }
            ]
        };
    } else if (recomputation === "full") {
        data = {
            name: "Activation Memory",
            children: [
                { name: 'LayerInput', value: s * b * h * bytesPerValue * L },
                { name: 'Dropout', value: inputDropout },
                { name: 'LayerNorm', value: outputLayerNorm },
                { name: 'Projection', value: outputLayerProjection },
                { name: 'Cross Entropy', value: outputCrossEntropy }
            ]
        };
    } else {
        throw new Error("Invalid recomputation value");
    }

    return data;
}

export function paramGradsOpt(h, L, s, v, k = 8, dp = 1, zero = 0, mixed = true) {
    // h, # hidden dimension size
    // L, # number of layers
    // s, # sequence length
    // v, # vocab size
    // k=8, # parameters for optimizer (Adam: 8 = 4 bytes moments + 4 bytes variance)
    // dp=1, # data parallelism
    // zero = 0, 1, 2, 3, # zero data parallelism
    // mixed=True # mixed precision training
    console.log('paramGradsOpt called with:', { h, L, s, v, k, dp, zero, mixed });
    const emb = h * (v + s);
    const oneLayer = 12 * h ** 2 + 13 * h;
    const other = 2 * h;

    const n = emb + L * oneLayer + other;

    if (mixed) {
        k += 4;
    }
    const bytesPerParameter = mixed ? 2 : 4;

    const data = {
        name: "Parameters / Gradients / Optimizer States",
        children: [
            { name: 'Parameters', value: zero >= 3 ? bytesPerParameter * n / dp : bytesPerParameter * n },
            { name: 'Gradients', value: zero >= 2 ? bytesPerParameter * n / dp : bytesPerParameter * n },
            { name: 'OptimizerAverages', value: zero >= 1 ? k * n / dp : k * n }
        ]
    };
    console.log('paramGradsOpt result:', data);
    return data;
}

export function updateGraph() {
    console.log('updateGraph called');
    const a = +document.getElementById('a').value;
    const b = +document.getElementById('b').value;
    const h = +document.getElementById('h').value;
    const h_ff = +document.getElementById('h_ff').value;
    const L = +document.getElementById('L').value;
    const s = +document.getElementById('s').value;
    const v = +document.getElementById('v').value;
    const k = +document.getElementById('k').value;
    const tp = +document.getElementById('tp').value;  // New: t parameter
    const zero = document.getElementById('zero').value;
    const dp = document.getElementById('dp').value;
    const mixed = document.getElementById('mixed').checked;
    const recomputation = document.getElementById('recomputation').value;
    const ff_activation = document.getElementById('ff_activation').value;
    const seq_parallel = document.getElementById('seq_parallel').checked;

    console.log('Slider values:', { a, b, h, h_ff, L, s, v, k, tp, zero, dp, mixed, recomputation, ff_activation, seq_parallel });

    const activationMemoryData = activationMemory(a, b, h, h_ff, L, s, v, tp, mixed, recomputation, ff_activation, seq_parallel);
    const paramGradsOptValue = paramGradsOpt(h, L, s, v, k, dp, zero, mixed);

    const data = {
        name: "root",
        children: [
            {
                name: 'Total',
                value: 0,
                children: [
                    activationMemoryData,
                    paramGradsOptValue
                ]
            }
        ]
    };

    console.log('Data for treemap:', data);

    const width = 600;
    const height = 600;
    const legendHeight = 50;

    const svg = d3.select("#graph").select("svg");
    svg.selectAll("*").remove();
    svg.attr("viewBox", [0, 0, width, height + legendHeight]);

    const treemap = d3.treemap()
        .size([width, height])
        .paddingOuter(3)
        .paddingTop(19)
        .paddingInner(3)
        .round(true);

    const root = d3.hierarchy(data)
        .sum(d => d.value);
    // .sort((a, b) => b.value - a.value);

    // const fixedSize100GB = 100 * 1024 * 1024 * 1024; // 100GB in bytes
    // if (root.children[0].value < fixedSize100GB) {
    //     root.value = fixedSize100GB;
    //     root.children[0].value = fixedSize100GB;
    // }

    console.log('Treemap root:', root);

    treemap(root);

    const color = d => {
        switch (d.data.name) {
            // Root and Total (container levels)
            case 'root': return 'rgb(225, 225, 225)';  // Light Grey
            case 'Total': return 'rgb(225, 225, 225)';  // Light Grey
            
            // Give distinct colors to the main section containers
            case 'Activation Memory': return 'rgb(61, 198, 159)';  // Orange
            case 'Parameters / Gradients / Optimizer States': return 'rgba(232, 137, 170, 0.85)';  // Teal Blue
    
            // Parameters / Gradients / Optimizer States branch
            case 'Parameters': return 'rgb(206, 192, 250)';  // Blue
            case 'Gradients': return 'rgb(227, 138, 66)';   // Orange
            case 'OptimizerAverages': return 'rgb(78, 165, 183)';  // Pink
            
            // activationMemory branch - Layer components
            case 'Attention': return 'rgb(206, 192, 250)';  // Purple
            case 'Feedforward': return 'rgb(171, 232, 241)';  // Light Blue
            case 'LayerNorm': return 'rgb(232, 137, 171)';  // Light Green
            
            // activationMemory branch - other components
            case 'Dropout': return 'rgb(67, 145, 108)';  // Dark Green
            case 'Projection': return 'rgb(174, 214, 251)';  // Sky Blue
            case 'Cross Entropy': return 'rgb(232, 137, 171)';  // Pink
            
            // Default for any Layer nodes and unexpected cases
            default: return 'rgb(227, 138, 66)';  // Light Grey
        };
      };

    if (d3.select('#tooltip').empty()) {
      d3.select('body')
      .append('div')
      .attr('id', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background-color', 'white')
      .style('padding', '4px')
      .style('font-size', '12px')
      .style('border-radius', '5px')
      .style('box-shadow', '0px 0px 5px 0px rgba(0,0,0,0.3)');
    }

    const cell = svg.selectAll("g")
        .data(root.descendants().filter(d => d.depth !== 0)) // Skip root node
        .join("g")
        .attr("transform", d => `translate(${d.x0},${d.y0})`)
        .on('mouseover', (event, d) => {
          const name = d.data.name;
          const value = formatBytes(d.value);
          d3.select('#tooltip').transition().duration(200).text(`${name}: ${value}`)
        })
        .on('mouseout', function() {
          d3.select('#tooltip').style('opacity', 0)
        })
        .on('mousemove', function(event) {
          d3.select('#tooltip').style('left', (event.pageX + 10) + 'px').style('top', (event.pageY + 10) + 'px').style('opacity', 1)
        });

    cell.append("rect")
        .attr("width", d => d.x1 - d.x0)
        .attr("height", d => d.y1 - d.y0)
        .attr("fill", d => color(d))
        .attr("stroke", d => d.depth === 1 ? color(d) : "white")
        .attr("stroke-width", 1);

    const fontSize = 10;
    const padding = 2;

    cell.append("text")
        .attr("font-size", `${fontSize}px`)
        .attr("font-family", "sans-serif")
        .each(function (d) {

            const node = d3.select(this);

            const name = d.data.name;
            const value = formatBytes(d.value);

            if (d.depth === 1 || d.depth === 2) {
                node.attr("transform", `translate(${padding},${fontSize + padding})`)
                    .attr("font-weight", "bold")
                    .attr("font-size", 12)
                    .text(`${name}: ${value}`);
            } else {
                // Child nodes
                node.attr("transform", `translate(${padding},${fontSize + padding})`)
                    .text(name[0].toUpperCase())  // Display only the first letter
                    .attr("font-weight", "bold")
                    .append("title")  // Add title for hover effect
                    .text(`${name}: ${value}`);
            }
        });

    /* 
    // Adjust legend positioning
    const legendData = root.children[0].children.concat(root.children[0]);
    const legend = svg.append("g")
        .attr("font-family", "sans-serif")
        .attr("font-size", 10)
        .attr("text-anchor", "start")
        .attr("transform", `translate(0, ${height})`)
        .selectAll("g")
        .data(legendData)
        .join("g")
        .attr("transform", (d, i) => `translate(${i * 240}, 0)`);

    legend.append("rect")
        .attr("x", 0)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", d => color(d))
        .attr("stroke", '#f3f3f3')
        .attr("stroke-width", 0);

    legend.append("text")
        .attr("x", 24)
        .attr("y", 9.5)
        .attr("dy", "0.32em")
        .text(d => `${d.data.name}: ${formatBytes(d.value)}`);
    */
    console.log('Treemap nodes created');
}
    
function formatBytes(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
    if (bytes === 0) return '0 Bytes';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)), 10);
    return `${(bytes / (1024 ** i)).toFixed(2)} ${sizes[i]}`;
}

const presets = {
    "Llama 3 Tiny": { a: 16, b: 3, h: 1024, h_ff: 4096, L: 1, s: 7, v: 30522, k: 8, tp: 1, zero: "1", dp: 1, mixed: true, recomputation: "none", ff_activation: "gelu", seq_parallel: false },
    "Llama 3 8B": { a: 32, b: 32, h: 4096, h_ff: 16384, L: 32, s: 256, v: 30522, k: 8, tp: 1, zero: "1", dp: 1, mixed: true, recomputation: "none", ff_activation: "swiglu", seq_parallel: false },
    "Llama 3 70B": { a: 64, b: 32, h: 8192, h_ff: 32768, L: 80, s: 256, v: 30522, k: 8, tp: 8, zero: "1", dp: 8, mixed: true, recomputation: "none", ff_activation: "swiglu", seq_parallel: false },
    "Llama 3 405B": { a: 128, b: 32, h: 16384, h_ff: 65536, L: 126, s: 256, v: 30522, k: 8, tp: 8, zero: "1", dp: 8, mixed: true, recomputation: "none", ff_activation: "swiglu", seq_parallel: false }
};

function setPresetValues(preset) {
    if (preset === "custom") return;

    const values = presets[preset];
    Object.keys(values).forEach(key => {
        const element = document.getElementById(key);
        const inputElement = document.getElementById(`${key}_input`);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = values[key];
            } else {
                element.value = values[key];
            }
        }
        if (inputElement) {
            inputElement.value = values[key];
        }
    });

    updateGraph();  // Add this line to ensure the graph updates when a preset is selected
}

function syncSliderAndInput(sliderId, inputId) {
    const slider = document.getElementById(sliderId);
    const input = document.getElementById(inputId);

    slider.addEventListener('input', () => {
        input.value = slider.value;
        updateGraph();
    });

    input.addEventListener('input', () => {
        let value = parseInt(input.value);
        if (isNaN(value)) {
            value = parseInt(slider.min);
        }
        value = Math.max(parseInt(slider.min), Math.min(parseInt(slider.max), value));
        slider.value = value;
        input.value = value;
        updateGraph();
    });
}

export const init_memory_plot = function () {
    console.log('Initializing memory plot');

    const sliderIds = ['a', 'b', 'h', 'h_ff', 'L', 's', 'v', 'k', 'tp', 'dp'];
    sliderIds.forEach(id => {
        const slider = document.getElementById(id);
        const input = document.getElementById(`${id}_input`);
        if (slider && input) {
            syncSliderAndInput(id, `${id}_input`);
        } else {
            console.warn(`Elements for ${id} not found`);
        }
    });

    const recomputationSelect = document.getElementById('recomputation');
    if (recomputationSelect) {
        recomputationSelect.addEventListener('change', updateGraph);
    } else {
        console.warn('Recomputation select not found');
    }

    const ffActivationSelect = document.getElementById('ff_activation');
    if (ffActivationSelect) {
        ffActivationSelect.addEventListener('change', updateGraph);
    } else {
        console.warn('FF Activation select not found');
    }

    const zeroSelect = document.getElementById('zero');
    if (zeroSelect) {
        zeroSelect.addEventListener('change', updateGraph);
    } else {
        console.warn('Zero select not found');
    }

    const mixedCheckbox = document.getElementById('mixed');
    if (mixedCheckbox) {
        mixedCheckbox.addEventListener('change', updateGraph);
    } else {
        console.warn('Mixed checkbox not found');
    }

    const seqParallelCheckbox = document.getElementById('seq_parallel');
    if (seqParallelCheckbox) {
        seqParallelCheckbox.addEventListener('change', updateGraph);
    } else {
        console.warn('Seq Parallel checkbox not found');
    }

    const presetSelect = document.getElementById('presets');
    if (presetSelect) {
        presetSelect.addEventListener('change', (event) => {
            setPresetValues(event.target.value);
        });
    } else {
        console.warn('Preset select not found');
    }

    // Set max values for sliders
    sliderIds.forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            switch (id) {
                case 'a': slider.max = '128'; break;
                case 'b': slider.max = '53248'; break;
                case 'h': slider.max = '16384'; break;
                case 'h_ff': slider.max = '65536'; break;
                case 'L': slider.max = '126'; break;
                case 's': slider.max = '128000'; break;
                case 'v': slider.max = '100000'; break;
                case 'k': slider.max = '16'; break;
                case 'tp': slider.max = '16'; break;
                case 'dp': slider.max = '256'; break;
            }
        } else {
            console.warn(`Slider ${id} not found`);
        }
    });

    console.log('Adding svg');
    const graphContainer = document.getElementById('graph');
    if (graphContainer) {
        const svg = d3.select("#graph")
            .append("svg")
    } else {
        console.warn('Graph container not found');
    }

    updateGraph();
};