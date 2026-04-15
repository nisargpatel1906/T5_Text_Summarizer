document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js Shader Background
    if (window.THREE) {
        initThreeJS();
    }

    const sourceText = document.getElementById('source-text');
    const wordCount = document.getElementById('word-count');
    const summarizeBtn = document.getElementById('summarize-btn');
    const clearBtn = document.getElementById('clear-btn');
    const maxLengthInput = document.getElementById('max-length');
    const maxLengthVal = document.getElementById('max-length-val');
    
    // Output elements
    const emptyState = document.getElementById('empty-state');
    const shimmer = document.getElementById('shimmer');
    const summaryResult = document.getElementById('summary-result');
    const copyBtn = document.getElementById('copy-btn');
    const spinner = document.querySelector('.loader-spinner');
    const btnText = document.querySelector('.btn-text');
    const btnIcon = document.querySelector('.fa-wand-magic-sparkles');

    // Update word count
    sourceText.addEventListener('input', (e) => {
        const text = e.target.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        wordCount.textContent = `${words} word${words !== 1 ? 's' : ''}`;
    });

    // Update range display
    maxLengthInput.addEventListener('input', (e) => {
        maxLengthVal.textContent = e.target.value;
    });
    
    // Clear button
    clearBtn.addEventListener('click', () => {
        sourceText.value = '';
        wordCount.textContent = '0 words';
        sourceText.focus();
    });

    // Handle summarize
    summarizeBtn.addEventListener('click', async () => {
        const text = sourceText.value.trim();
        
        if (!text) {
            // Shake animation for error
            sourceText.parentElement.style.animation = 'shake 0.4s ease';
            setTimeout(() => sourceText.parentElement.style.animation = '', 400);
            return;
        }

        // Set Loading state
        summarizeBtn.disabled = true;
        btnText.textContent = 'Processing...';
        btnIcon.classList.add('hidden');
        spinner.classList.remove('hidden');
        
        emptyState.classList.add('hidden');
        summaryResult.classList.add('hidden');
        copyBtn.classList.add('hidden');
        shimmer.classList.remove('hidden');
        
        try {
            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    max_length: parseInt(maxLengthInput.value)
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to summarize text');
            }

            // Display result
            summaryResult.innerHTML = `<p>${data.summary}</p>`;
            
        } catch (error) {
            console.error('Error:', error);
            summaryResult.innerHTML = `<p style="color: #ef4444;"><i class="fa-solid fa-circle-exclamation"></i> Error: ${error.message}</p>`;
        } finally {
            // Remove loading state
            summarizeBtn.disabled = false;
            btnText.textContent = 'Generate Summary';
            btnIcon.classList.remove('hidden');
            spinner.classList.add('hidden');
            
            shimmer.classList.add('hidden');
            summaryResult.classList.remove('hidden');
            copyBtn.classList.remove('hidden');
        }
    });

    // Handle copy
    copyBtn.addEventListener('click', () => {
        const textToCopy = summaryResult.innerText;
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyBtn.innerHTML = '<i class="fa-solid fa-check" style="color: #10b981;"></i>';
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
            }, 2000);
        });
    });
});

// Three.js Shader Background
function initThreeJS() {
    const container = document.getElementById('shader-bg');
    if (!container) return;

    const THREE = window.THREE;
    const camera = new THREE.Camera();
    camera.position.z = 1;
    const scene = new THREE.Scene();
    const geometry = new THREE.PlaneBufferGeometry(2, 2);

    const uniforms = {
        time: { type: "f", value: 1.0 },
        resolution: { type: "v2", value: new THREE.Vector2() }
    };

    const vertexShader = `
      void main() {
        gl_Position = vec4( position, 1.0 );
      }
    `;

    const fragmentShader = `
      #define TWO_PI 6.2831853072
      #define PI 3.14159265359

      precision highp float;
      uniform vec2 resolution;
      uniform float time;
        
      float random (in float x) {
          return fract(sin(x)*1e4);
      }
      float random (vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898,78.233)))* 43758.5453123);
      }
      
      varying vec2 vUv;

      void main(void) {
        vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
        
        vec2 fMosaicScal = vec2(2.0, 4.0);
        vec2 vScreenSize = vec2(256.0, 256.0);
        uv.x = floor(uv.x * vScreenSize.x / fMosaicScal.x) / (vScreenSize.x / fMosaicScal.x);
        uv.y = floor(uv.y * vScreenSize.y / fMosaicScal.y) / (vScreenSize.y / fMosaicScal.y);       
          
        float t = time*0.06+random(uv.y)*0.4;
        float lineWidth = 0.0008;

        vec3 color = vec3(0.0);
        for(int j = 0; j < 3; j++){
          for(int i=0; i < 5; i++){
            color[j] += lineWidth*float(i*i) / abs(fract(t - 0.01*float(j)+float(i)*0.01)*1.0 - length(uv));        
          }
        }

        gl_FragColor = vec4(color[2],color[1],color[0],1.0);
      }
    `;

    const material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const onWindowResize = () => {
        const rect = container.getBoundingClientRect();
        renderer.setSize(rect.width, rect.height);
        uniforms.resolution.value.x = renderer.domElement.width;
        uniforms.resolution.value.y = renderer.domElement.height;
    };
    onWindowResize();
    window.addEventListener("resize", onWindowResize, false);

    const animate = () => {
        requestAnimationFrame(animate);
        uniforms.time.value += 0.05;
        renderer.render(scene, camera);
    };
    animate();
}
