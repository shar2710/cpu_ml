document.addEventListener('DOMContentLoaded', function() {
    const simulationForm = document.getElementById('simulation_form');
    if (!simulationForm) return; 
    initFormValues();
    
   
    setupFormListeners();
    
    // Add form validation
    setupFormValidation();
});

// Initialize form with default values
function initFormValues() {
    // Make sure the form exists (already checked in DOMContentLoaded, but double-checking for safety)
    const simulationForm = document.getElementById('simulation_form');
    if (!simulationForm) return;
    const latencyWeightSlider = document.getElementById('reward_latency_weight_slider');
    const latencyWeightInput = document.getElementById('reward_latency_weight');
    
    if (latencyWeightSlider && latencyWeightInput) {
        latencyWeightSlider.value = latencyWeightInput.value;
    }
    
    // Update sliders and dependent fields
    updateSliderValues();
    
    // Update workload description based on the selected value
    updateWorkloadDescription();
}

// Set up event listeners for form elements
function setupFormListeners() {
    // Only run this function on pages with the simulation form
    const simulationForm = document.getElementById('simulation_form');
    if (!simulationForm) return;
    
    // Listen for workload type changes
    const workloadTypeSelect = document.getElementById('workload_type');
    if (workloadTypeSelect) {
        workloadTypeSelect.addEventListener('change', updateWorkloadDescription);
    }
    
    // Listen for slider changes
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        slider.addEventListener('input', updateSliderValues);
    });
    
    // Listen for number input changes
    const numInputs = document.querySelectorAll('input[type="number"]');
    numInputs.forEach(input => {
        input.addEventListener('change', function() {
            // Find corresponding slider if exists and update it
            const sliderId = input.id + '_slider';
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.value = input.value;
            }
        });
    });
    
    // Special case for reward weights
    const latencyWeightSlider = document.getElementById('reward_latency_weight_slider');
    const latencyWeightInput = document.getElementById('reward_latency_weight');
    const throughputWeightInput = document.getElementById('reward_throughput_weight');
    const rewardRatioDisplay = document.getElementById('reward_ratio_display');
    
    if (latencyWeightSlider && latencyWeightInput && throughputWeightInput && rewardRatioDisplay) {
        latencyWeightSlider.addEventListener('input', function() {
            const latencyWeight = parseFloat(this.value);
            const throughputWeight = 1 - latencyWeight;
            
            latencyWeightInput.value = latencyWeight.toFixed(2);
            throughputWeightInput.value = throughputWeight.toFixed(2);
            
            // Update visual indicator
            const latencyPercentage = Math.round(latencyWeight * 100);
            rewardRatioDisplay.textContent = 
                `${latencyPercentage}% Latency / ${100 - latencyPercentage}% Throughput`;
        });
    }
}

// Update all slider value displays
function updateSliderValues() {
    // Update simple sliders
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const outputId = slider.id + '_value';
        const output = document.getElementById(outputId);
        if (output) {
            output.textContent = slider.value;
        }
    });
    
    // Special case for reward weights
    const latencyWeightSlider = document.getElementById('reward_latency_weight_slider');
    if (!latencyWeightSlider) return;
    
    const latencyWeightInput = document.getElementById('reward_latency_weight');
    const throughputWeightInput = document.getElementById('reward_throughput_weight');
    const rewardRatioDisplay = document.getElementById('reward_ratio_display');
    
    // Make sure all required elements exist before continuing
    if (!latencyWeightInput || !throughputWeightInput || !rewardRatioDisplay) return;
    
    const latencyWeight = parseFloat(latencyWeightSlider.value);
    const throughputWeight = 1 - latencyWeight;
    
    latencyWeightInput.value = latencyWeight.toFixed(2);
    throughputWeightInput.value = throughputWeight.toFixed(2);
    
    // Update visual indicator
    const latencyPercentage = Math.round(latencyWeight * 100);
    rewardRatioDisplay.textContent = 
        `${latencyPercentage}% Latency / ${100 - latencyPercentage}% Throughput`;
}

// Update workload description based on selected type
function updateWorkloadDescription() {
    const workloadTypeElement = document.getElementById('workload_type');
    const descriptionElement = document.getElementById('workload_description');
    
    // Return early if either element doesn't exist on the current page
    if (!workloadTypeElement || !descriptionElement) return;
    
    const workloadType = workloadTypeElement.value;
    let description = '';
    
    switch (workloadType) {
        case 'trading':
            description = 'High-frequency trading workload with a mix of short, latency-sensitive tasks and longer analysis tasks.';
            break;
        case 'realtime':
            description = 'Real-time system workload with periodic tasks, strict deadlines, and consistent execution times.';
            break;
        case 'mixed':
            description = 'A mixed workload combining trading and real-time characteristics with diverse task requirements.';
            break;
        default:
            description = 'Select a workload type to see its description.';
    }
    
    descriptionElement.textContent = description;
}

// Set up form validation
function setupFormValidation() {
    const simulationForm = document.getElementById('simulation_form');
    if (!simulationForm) return;
    
    simulationForm.addEventListener('submit', function(event) {
        // Get form elements
        const numCpusEl = document.getElementById('num_cpus');
        const numTasksEl = document.getElementById('num_tasks');
        const learningRateEl = document.getElementById('learning_rate');
        const gammaEl = document.getElementById('gamma');
        const epsilonEl = document.getElementById('epsilon');
        
        // Check if all elements exist
        if (!numCpusEl || !numTasksEl || !learningRateEl || !gammaEl || !epsilonEl) {
            console.warn('Form validation: Some form elements are missing');
            return;
        }
        
        // Basic validation
        const numCpus = parseInt(numCpusEl.value);
        const numTasks = parseInt(numTasksEl.value);
        
        let isValid = true;
        let errorMessage = '';
        
        if (numCpus <= 0 || numCpus > 16) {
            errorMessage = 'Number of CPUs must be between 1 and 16.';
            isValid = false;
        } else if (numTasks <= 0 || numTasks > 1000) {
            errorMessage = 'Number of tasks must be between 1 and 1000.';
            isValid = false;
        }
        
        // Advanced validation for RL parameters
        const learningRate = parseFloat(learningRateEl.value);
        const gamma = parseFloat(gammaEl.value);
        const epsilon = parseFloat(epsilonEl.value);
        
        if (learningRate <= 0 || learningRate > 1) {
            errorMessage = 'Learning rate must be between 0 and 1.';
            isValid = false;
        } else if (gamma <= 0 || gamma > 1) {
            errorMessage = 'Discount factor (gamma) must be between 0 and 1.';
            isValid = false;
        } else if (epsilon < 0 || epsilon > 1) {
            errorMessage = 'Exploration rate (epsilon) must be between 0 and 1.';
            isValid = false;
        }
        
        if (!isValid) {
            event.preventDefault();
            
            // Show error message
            const errorAlert = document.getElementById('form_error');
            if (errorAlert) {
                errorAlert.textContent = errorMessage;
                errorAlert.classList.remove('d-none');
                
                // Scroll to error
                errorAlert.scrollIntoView({behavior: 'smooth'});
            }
        }
    });
}

// Function to update the training progress bar
function updateTrainingProgress(progress) {
    const progressBar = document.getElementById('training_progress');
    const progressText = document.getElementById('progress_text');
    
    if (progressBar && progressText) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        progressText.textContent = `${progress}%`;
    }
}

// Initialize training simulation
function startTraining() {
    const numEpisodesEl = document.getElementById('num_episodes');
    const trainingStatus = document.getElementById('training_status');
    
    // Make sure the elements exist
    if (!numEpisodesEl || !trainingStatus) {
        console.warn('startTraining: Required elements not found');
        return;
    }
    
    const numEpisodes = parseInt(numEpisodesEl.value) || 100; // Default to 100 if parsing fails
    
    trainingStatus.classList.remove('d-none');
    
    // Simulate training progress
    let progress = 0;
    const interval = numEpisodes > 0 ? (100 / numEpisodes) : 10; // Avoid division by zero
    
    const progressTimer = setInterval(() => {
        progress += interval;
        if (progress >= 100) {
            progress = 100;
            clearInterval(progressTimer);
            
            // Enable the view results button
            const resultsButton = document.getElementById('view_results_btn');
            if (resultsButton) {
                resultsButton.disabled = false;
            }
        }
        
        updateTrainingProgress(Math.round(progress));
    }, 500);
}
