class StatsManager {
    constructor() {
        this.frameCount = 0;
        this.totalBytes = 0;
        this.bytesInLastSecond = 0;
        this.lastUpdate = performance.now();
        this.bandwidthData = {
            time: [],
            value: []
        };

        // 初始化图表
        this.initChart();
    }

    initChart() {
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 30, r: 30, b: 30, l: 50 },
            font: { color: '#fff' },
            showlegend: false,
            yaxis: {
                title: 'KB/s',
                gridcolor: '#333',
                range: [0, 1000]
            },
            xaxis: {
                gridcolor: '#333',
                showticklabels: false
            }
        };

        Plotly.newPlot('bandwidthChart', [{
            y: this.bandwidthData.value,
            type: 'line',
            line: { color: '#4CAF50' }
        }], layout);
    }

    updateStats(bytes) {
        const now = performance.now();
        this.totalBytes += bytes;
        this.bytesInLastSecond += bytes;
        this.frameCount++;

        if (now - this.lastUpdate >= 1000) {
            const elapsed = (now - this.lastUpdate) / 1000;
            const bytesPerSecond = this.bytesInLastSecond / elapsed;
            const fps = this.frameCount / elapsed;

            this.updateDisplay(bytesPerSecond, fps);
            this.updateChart(bytesPerSecond);

            // 重置计数器
            this.bytesInLastSecond = 0;
            this.frameCount = 0;
            this.lastUpdate = now;
        }
    }

    updateDisplay(bytesPerSecond, fps) {
        document.getElementById('bandwidth').textContent = 
            `${(bytesPerSecond / 1024).toFixed(2)} KB/s`;
        document.getElementById('totalData').textContent = 
            `${(this.totalBytes / (1024 * 1024)).toFixed(2)} MB`;
        document.getElementById('fps').textContent = 
            `${fps.toFixed(1)} FPS`;
    }

    updateChart(bytesPerSecond) {
        this.bandwidthData.value.push(bytesPerSecond / 1024);
        if (this.bandwidthData.value.length > 30) {
            this.bandwidthData.value.shift();
        }
        
        Plotly.update('bandwidthChart', {
            y: [this.bandwidthData.value]
        });
    }
} 