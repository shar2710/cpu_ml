function createComparisonChart(elementId, title, data, yAxisLabel) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Extract labels and data series
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    // Generate colors for each scheduler
    const colors = [
        'rgba(54, 162, 235, 0.8)',  // Blue for RL
        'rgba(255, 99, 132, 0.8)',  // Red
        'rgba(75, 192, 192, 0.8)',  // Green
        'rgba(255, 159, 64, 0.8)'   // Orange
    ];
    
    // Create chart
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: yAxisLabel,
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: colors.slice(0, labels.length).map(color => color.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: yAxisLabel
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Scheduler'
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const label = labels[index];
                    highlightScheduler(label);
                }
            }
        }
    });
}

// Create a radar chart to compare multiple metrics
function createRadarChart(elementId, data) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Extract scheduler names
    const schedulers = Object.keys(data.latency);
    
    // Prepare datasets
    const datasets = schedulers.map((scheduler, index) => {
        const colors = [
            'rgba(54, 162, 235, 0.2)',  // Blue for RL
            'rgba(255, 99, 132, 0.2)',  // Red
            'rgba(75, 192, 192, 0.2)',  // Green
            'rgba(255, 159, 64, 0.2)'   // Orange
        ];
        
        const borderColors = [
            'rgb(54, 162, 235)',       // Blue for RL
            'rgb(255, 99, 132)',       // Red
            'rgb(75, 192, 192)',       // Green
            'rgb(255, 159, 64)'        // Orange
        ];
        
        // Invert latency so lower is better for all metrics
        const latencyValue = 1 - (data.latency[scheduler] / Math.max(...Object.values(data.latency)));
        
        return {
            label: scheduler,
            data: [
                latencyValue,
                data.throughput[scheduler],
                data.utilization[scheduler]
            ],
            backgroundColor: colors[index % colors.length],
            borderColor: borderColors[index % borderColors.length],
            borderWidth: 2,
            pointBackgroundColor: borderColors[index % borderColors.length],
            pointRadius: 4
        };
    });
    
    // Create radar chart
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Latency (Lower Better)', 'Throughput', 'CPU Utilization'],
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Scheduler Performance Comparison',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                r: {
                    min: 0,
                    max: 1,
                    ticks: {
                        display: false
                    }
                }
            }
        }
    });
}

// Create a line chart for reward history
function createRewardHistoryChart(elementId, rewardData) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    // Create array of episode numbers
    const episodes = Array.from({length: rewardData.length}, (_, i) => i + 1);
    
    // Create chart
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: episodes,
            datasets: [{
                label: 'Reward',
                data: rewardData,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Rewards Over Episodes',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Reward'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Episode'
                    }
                }
            }
        }
    });
}

// Highlight a specific scheduler's results
function highlightScheduler(schedulerName) {
    // Remove existing highlights
    document.querySelectorAll('.highlighted').forEach(el => {
        el.classList.remove('highlighted');
    });
    
    // Add highlight to the selected scheduler
    document.querySelectorAll(`.scheduler-${schedulerName}`).forEach(el => {
        el.classList.add('highlighted');
    });
}

// Fetch metrics data from API and update charts
function loadMetricsData() {
    fetch('/api/compare_metrics')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error(data.error);
                return;
            }
            
            // Create comparison charts
            createComparisonChart('latencyChart', 'Average Latency Comparison', data.latency, 'Latency (ms)');
            createComparisonChart('throughputChart', 'Throughput Comparison', data.throughput, 'Tasks/Second');
            createComparisonChart('utilizationChart', 'CPU Utilization Comparison', data.utilization, 'Utilization (%)');
            
            // Create radar chart
            createRadarChart('radarChart', data);
        })
        .catch(error => {
            console.error('Error fetching metrics data:', error);
        });
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the results page
    if (document.getElementById('latencyChart')) {
        loadMetricsData();
    }
    
    // Load training rewards chart if exists
    const rewardsChartElement = document.getElementById('rewardsChart');
    if (rewardsChartElement && rewardsChartElement.dataset.rewards) {
        const rewardsData = JSON.parse(rewardsChartElement.dataset.rewards);
        createRewardHistoryChart('rewardsChart', rewardsData);
    }
});
