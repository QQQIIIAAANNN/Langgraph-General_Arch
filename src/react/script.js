    // Ensure mainModelPath is defined (passed from template)
    if (typeof mainModelPath !== 'undefined' && mainModelPath) {
        const container = document.getElementById('model-viewer-container');
        if (container) {
            let scene, camera, renderer, controls, model;

            function initThree() {
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x333333); // Darker background might help see unlit materials better initially
                // scene.fog = new THREE.Fog(0x333333, 10, 30); // Optional: add fog

                camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.set(0, 1.5, 4); // Slightly adjusted camera

                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                
                // Color Space - Important for GLTF
                // renderer.outputColorSpace = THREE.SRGBColorSpace; // From r152, this is default. For older, use sRGBEncoding.
                // For versions < r152, you might need:
                // renderer.outputEncoding = THREE.sRGBEncoding; 


                container.appendChild(renderer.domElement);

                // Enhanced Lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.0); // Stronger ambient
                scene.add(ambientLight);
                
                const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.5); // Stronger directional
                directionalLight1.position.set(5, 10, 7.5).normalize();
                scene.add(directionalLight1);

                const directionalLight2 = new THREE.DirectionalLight(0xffffff, 1.0); // Another directional
                directionalLight2.position.set(-5, -5, -10).normalize();
                scene.add(directionalLight2);

                // Optional: Hemisphere light for softer ambient lighting
                // const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
                // hemisphereLight.position.set(0, 20, 0);
                // scene.add(hemisphereLight);


                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.minDistance = 1;
                controls.maxDistance = 20;
                controls.target.set(0, 0.5, 0); // Adjust target if model is not centered at origin vertically
                controls.update();


                console.log("Attempting to load model from:", mainModelPath);
                const loader = new THREE.GLTFLoader();
                loader.load(
                    mainModelPath,
                    function (gltf) {
                        model = gltf.scene;
                        console.log("Model loaded successfully, processing...", model);

                        // Optional: Traverse and log material info
                        model.traverse(function (child) {
                            if (child.isMesh) {
                                console.log("Mesh found:", child.name, "Material:", child.material);
                                if (child.material) {
                                    // If you were using older Three.js and manual sRGB:
                                    // if (child.material.map) child.material.map.encoding = THREE.sRGBEncoding;
                                    // if (child.material.emissiveMap) child.material.emissiveMap.encoding = THREE.sRGBEncoding;
                                    // if (child.material.metalnessMap) child.material.metalnessMap.encoding = THREE.sRGBEncoding; // Not sRGB
                                    // if (child.material.roughnessMap) child.material.roughnessMap.encoding = THREE.sRGBEncoding; // Not sRGB
                                }
                            }
                        });
                        
                        const box = new THREE.Box3().setFromObject(model);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());
                        
                        model.position.sub(center); // Center the model at origin
                        // Optional: move model slightly up if its pivot is at its base
                        // model.position.y += size.y / 2; 

                        console.log("Model bounding box size:", size);
                        const maxDim = Math.max(size.x, size.y, size.z);
                        
                        if (maxDim > 0.001) { // Check if maxDim is valid (increased threshold slightly)
                            const desiredSize = 2.5; // Adjusted desired size
                            const scale = desiredSize / maxDim;
                            model.scale.set(scale, scale, scale);
                            console.log("Model scaled to:", scale);
                        } else {
                            console.warn("Model maxDim is very small or zero, using default scale (1,1,1). Model might be empty or too small.");
                            model.scale.set(1, 1, 1); 
                        }

                        scene.add(model);
                        console.log("Model added to scene.");
                        
                        // Adjust camera to look at the model after loading and scaling
                        // box.setFromObject(model); // Recompute box after scaling and positioning
                        // box.getCenter(controls.target); // Target the center of the scaled model
                        // camera.lookAt(controls.target); // Ensure camera looks at the target

                        animate();
                    },
                    function (xhr) { // onProgress callback
                        console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
                    },
                    function (error) {
                        console.error('Error loading model:', error);
                        const errorMsg = document.createElement('p');
                        errorMsg.textContent = `Error loading 3D model from ${mainModelPath}. Check console for details. Error: ${error.message || error}`;
                        errorMsg.style.color = 'red';
                        errorMsg.style.padding = '10px';
                        container.appendChild(errorMsg);
                    }
                );

                window.addEventListener('resize', onWindowResize, false);
            }

            function onWindowResize() {
                if (camera && renderer && container) {
                    camera.aspect = container.clientWidth / container.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, container.clientHeight);
                }
            }

            function animate() {
                requestAnimationFrame(animate);
                if (controls) controls.update(); // Only call if controls exist
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                } else {
                    // This case should ideally not happen if initThree completed
                    // console.warn("Renderer, scene, or camera not available for animation frame.");
                }
            }

            if (THREE && THREE.GLTFLoader && THREE.OrbitControls) {
                initThree();
            } else {
                console.error("Three.js or its components (GLTFLoader, OrbitControls) not loaded.");
                const errorMsg = document.createElement('p');
                errorMsg.textContent = "Error: Could not initialize 3D viewer. Required libraries missing.";
                container.appendChild(errorMsg);
            }

        } else {
            console.warn("Model viewer container not found in HTML.");
        }
    } else {
        console.warn("Main 3D model path not provided or empty. Skipping 3D viewer initialization.");
        const container = document.getElementById('model-viewer-container');
        if(container){
            const noModelMsg = document.createElement('p');
            noModelMsg.textContent = "No primary 3D model specified for display.";
            noModelMsg.style.textAlign = "center";
            noModelMsg.style.padding = "20px";
            container.appendChild(noModelMsg);
        }
    }

    // Add other UI interaction logic here (e.g., for task details visibility)
    document.addEventListener('DOMContentLoaded', () => {
        const taskCards = document.querySelectorAll('.task-card');
        taskCards.forEach(card => {
            const header = card.querySelector('h3');
            // Wrap content after h3 in a div if not already wrapped
            let currentElement = header.nextElementSibling;
            let contentWrapper = card.querySelector('.task-card-content');

            if (!contentWrapper) {
                contentWrapper = document.createElement('div');
                contentWrapper.classList.add('task-card-content');
                while (currentElement) {
                    let nextSibling = currentElement.nextElementSibling; // Store next before moving
                    contentWrapper.appendChild(currentElement);
                    currentElement = nextSibling;
                }
                header.insertAdjacentElement('afterend', contentWrapper);
            }

            // Initially hide the content
            if (contentWrapper) { // Check if wrapper exists
                 contentWrapper.style.display = 'none';
                 card.classList.add('collapsed');
            }


            if (header) { // Check if header exists
                header.style.cursor = 'pointer';
                header.addEventListener('click', () => {
                    if (contentWrapper) { // Check if wrapper exists for toggle
                        const isCollapsed = contentWrapper.style.display === 'none';
                        contentWrapper.style.display = isCollapsed ? 'block' : 'none';
                        card.classList.toggle('collapsed', !isCollapsed);
                    }
                });
            }
        });
    });
