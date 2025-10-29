// Initialize particles.js for background animation
document.addEventListener('DOMContentLoaded', function() {
    // Particles.js configuration
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: {
                    value: 80,
                    density: {
                        enable: true,
                        value_area: 800
                    }
                },
                color: {
                    value: '#0d6efd'
                },
                shape: {
                    type: 'circle',
                    stroke: {
                        width: 0,
                        color: '#000000'
                    }
                },
                opacity: {
                    value: 0.5,
                    random: true,
                    anim: {
                        enable: true,
                        speed: 1,
                        opacity_min: 0.1,
                        sync: false
                    }
                },
                size: {
                    value: 3,
                    random: true,
                    anim: {
                        enable: true,
                        speed: 2,
                        size_min: 0.1,
                        sync: false
                    }
                },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#0d6efd',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 1,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false,
                    attract: {
                        enable: false,
                        rotateX: 600,
                        rotateY: 1200
                    }
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: {
                        enable: true,
                        mode: 'grab'
                    },
                    onclick: {
                        enable: true,
                        mode: 'push'
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 140,
                        line_linked: {
                            opacity: 1
                        }
                    },
                    bubble: {
                        distance: 400,
                        size: 40,
                        duration: 2,
                        opacity: 8,
                        speed: 3
                    },
                    repulse: {
                        distance: 200,
                        duration: 0.4
                    },
                    push: {
                        particles_nb: 4
                    },
                    remove: {
                        particles_nb: 2
                    }
                }
            },
            retina_detect: true
        });
    }

    // File Upload Handling
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const previewImage = document.getElementById('preview-image');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');
    const uploadPreview = document.querySelector('.upload-preview');
    const submitBtn = document.getElementById('submit-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const submitText = document.getElementById('submit-text');
    const resultSection = document.getElementById('result-section');
    const tryAgainBtn = document.getElementById('try-again-btn');
    
    if (dropArea) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop area when file is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        // Handle dropped files
        dropArea.addEventListener('drop', handleDrop, false);
        
        // Handle file input change
        if (fileInput) {
            fileInput.addEventListener('change', handleFiles, false);
        }
        
        // Handle form submission
        if (uploadForm) {
            uploadForm.addEventListener('submit', handleSubmit, false);
        }
        
        // Try again button
        if (tryAgainBtn) {
            tryAgainBtn.addEventListener('click', resetForm, false);
        }
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('drag-over');
    }

    function unhighlight() {
        dropArea.classList.remove('drag-over');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(e) {
        let files;
        if (e instanceof FileList) {
            files = e;
        } else {
            files = e.target.files;
        }
        
        if (files && files.length) {
            const file = files[0];
            
            // Check if file is an image
            if (!file.type.match('image.*')) {
                showError('Please upload an image file (JPEG, PNG, or BMP)');
                return;
            }
            
            // Preview the image
            uploadPlaceholder.classList.add('d-none');
            uploadPreview.classList.remove('d-none');
            
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                
                // Enable submit button
                submitBtn.disabled = false;
                
                // Add animation to the preview
                previewImage.classList.add('animate__animated', 'animate__fadeIn');
            };
            reader.readAsDataURL(file);
        }
    }

    function handleSubmit(e) {
        e.preventDefault();
        
        // Show loading state
        submitBtn.disabled = true;
        loadingSpinner.classList.remove('d-none');
        submitText.textContent = 'Analyzing...';
        
        const formData = new FormData(uploadForm);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading state
            loadingSpinner.classList.add('d-none');
            submitText.textContent = 'Analyze Fingerprint';
            
            // Show results
            showResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loadingSpinner.classList.add('d-none');
            submitText.textContent = 'Analyze Fingerprint';
            submitBtn.disabled = false;
            
            showError('An error occurred. Please try again.');
        });
    }

    function showResults(data) {
        if (data.error) {
            showError(data.error);
            submitBtn.disabled = false;
            return;
        }
        
        // Hide upload section animation
        const uploadSection = document.getElementById('upload-section');
        uploadSection.classList.add('animate__fadeOutUp');
        
        setTimeout(() => {
            // Hide upload section
            uploadSection.style.display = 'none';
            
            // Update result values
            document.getElementById('result-blood-group').textContent = data.blood_group;
            document.getElementById('result-confidence').textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
            
            // Show blood group facts based on the predicted blood group
            showBloodGroupFacts(data.blood_group);
            
            // Show probability chart
            createProbabilityChart(data.all_probabilities);
            
            // Show result section
            resultSection.classList.remove('d-none');
            resultSection.classList.add('animate__fadeInUp');
        }, 500);
    }

    function createProbabilityChart(probabilities) {
        const ctx = document.createElement('canvas');
        ctx.id = 'probabilityChart';
        document.getElementById('probability-chart').innerHTML = '';
        document.getElementById('probability-chart').appendChild(ctx);
        
        const labels = Object.keys(probabilities);
        const data = Object.values(probabilities);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: data,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#f8f9fa'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#f8f9fa'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#f8f9fa'
                        }
                    }
                }
            }
        });
    }

    function showBloodGroupFacts(bloodGroup) {
        const factsContent = document.getElementById('blood-facts-content');
        const facts = getBloodGroupFacts(bloodGroup);
        
        let factsHTML = `
            <div class="blood-group-info">
                <h5>${bloodGroup} Blood Group</h5>
                <ul>
        `;
        
        facts.forEach(fact => {
            factsHTML += `<li>${fact}</li>`;
        });
        
        factsHTML += `
                </ul>
            </div>
        `;
        
        factsContent.innerHTML = factsHTML;
    }

    function getBloodGroupFacts(bloodGroup) {
        const facts = {
            'A+': [
                'Can receive blood from A+, A-, O+, and O- donors',
                'Can donate blood to A+ and AB+ recipients',
                'Approximately 34% of the population has this blood type'
            ],
            'A-': [
                'Can receive blood from A- and O- donors',
                'Can donate blood to A+, A-, AB+, and AB- recipients',
                'About 6% of the population has this blood type'
            ],
            'B+': [
                'Can receive blood from B+, B-, O+, and O- donors',
                'Can donate blood to B+ and AB+ recipients',
                'About 9% of the population has this blood type'
            ],
            'B-': [
                'Can receive blood from B- and O- donors',
                'Can donate blood to B+, B-, AB+, and AB- recipients',
                'Only about 2% of the population has this blood type'
            ],
            'AB+': [
                'Universal recipient - can receive blood from all blood types',
                'Can donate blood only to AB+ recipients',
                'Approximately 3% of the population has this blood type'
            ],
            'AB-': [
                'Can receive blood from A-, B-, AB-, and O- donors',
                'Can donate blood to AB+ and AB- recipients',
                'Rarest blood type, with less than 1% of the population'
            ],
            'O+': [
                'Can receive blood from O+ and O- donors',
                'Can donate blood to A+, B+, AB+, and O+ recipients',
                'About 38% of the population has this blood type'
            ],
            'O-': [
                'Universal donor - can donate blood to all blood types',
                'Can receive blood only from O- donors',
                'Approximately 7% of the population has this blood type'
            ]
        };
        
        return facts[bloodGroup] || [
            'Information not available for this blood group',
            'Please consult a healthcare professional for accurate information'
        ];
    }

    function resetForm() {
        // Hide result section
        resultSection.classList.add('animate__fadeOutDown');
        
        setTimeout(() => {
            resultSection.classList.remove('animate__fadeOutDown');
            resultSection.classList.add('d-none');
            
            // Reset form
            uploadForm.reset();
            
            // Reset UI
            uploadPlaceholder.classList.remove('d-none');
            uploadPreview.classList.add('d-none');
            submitBtn.disabled = true;
            
            // Show upload section
            const uploadSection = document.getElementById('upload-section');
            uploadSection.style.display = 'block';
            uploadSection.classList.remove('animate__fadeOutUp');
            uploadSection.classList.add('animate__fadeInDown');
        }, 500);
    }

    function showError(message) {
        Swal.fire({
            title: 'Error!',
            text: message,
            icon: 'error',
            confirmButtonText: 'OK',
            background: '#1e1e1e',
            color: '#f8f9fa',
            confirmButtonColor: '#0d6efd'
        });
    }
    
    // Add scroll animations
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.tech-item, .card');
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (elementPosition < screenPosition) {
                element.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    };
    
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Initial check on page load
}); 