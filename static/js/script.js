document.addEventListener('DOMContentLoaded', function() {
    // Manejo de pestañas
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            // Desactivar todas las pestañas
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Activar la pestaña seleccionada
            button.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });

    // Manejo de carga de archivos
    const fileInput = document.getElementById('file');
    const fileInfo = document.querySelector('.file-info');

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const fileName = this.files[0].name;
                fileInfo.textContent = fileName;
                fileInfo.style.color = 'var(--primary-color)';
            } else {
                fileInfo.textContent = 'Ningún archivo seleccionado';
                fileInfo.style.color = 'var(--gray-color)';
            }
        });
    }

    // Manejo de FAQ
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        
        question.addEventListener('click', () => {
            // Cerrar todos los otros items
            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });
            
            // Toggle el item actual
            item.classList.toggle('active');
        });
    });

    // Navegación suave
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
                
                // Actualizar clase active en la navegación
                document.querySelectorAll('.main-nav a').forEach(navLink => {
                    navLink.classList.remove('active');
                });
                
                document.querySelector(`.main-nav a[href="${targetId}"]`)?.classList.add('active');
            }
        });
    });

    // Actualizar navegación al hacer scroll
    window.addEventListener('scroll', function() {
        let currentSection = '';
        const sections = document.querySelectorAll('section[id]');
        const scrollPosition = window.scrollY;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 150;
            const sectionHeight = section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                currentSection = '#' + section.getAttribute('id');
            }
        });
        
        document.querySelectorAll('.main-nav a').forEach(navLink => {
            navLink.classList.remove('active');
            if (navLink.getAttribute('href') === currentSection) {
                navLink.classList.add('active');
            }
        });
    });

    // Manejar predicción individual
const individualForm = document.getElementById('individual-form');
if (individualForm) {
    individualForm.addEventListener('submit', async function(e) {
        e.preventDefault(); // Evitar recargar la página
        const form = e.target;
        const formData = new FormData(form);
        const params = new URLSearchParams();
        
        for (const pair of formData) {
            params.append(pair[0], pair[1]);
        }

        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<div class="loading">Procesando predicción</div>';
        resultDiv.className = 'result-box';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: params
            });

            if (response.ok) {
                const data = await response.json();
                resultDiv.innerHTML = `
                    <div class="success">
                        <h3><i class="fas fa-check-circle"></i> Predicción Completada</h3>
                        <p><strong>Resultado:</strong> ${data.message}</p>
                    </div>
                `;
            } else {
                const error = await response.json();
                resultDiv.innerHTML = `
                    <div class="error">
                        <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
                        <p>${error.error || 'Ha ocurrido un error al procesar la solicitud'}</p>
                    </div>
                `;
            }
        } catch (err) {
            resultDiv.innerHTML = `
                <div class="error">
                    <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
                    <p>Error al procesar la solicitud: ${err.message}</p>
                </div>
            `;
        }
    });
}

    // Manejar predicción por lotes
    const batchForm = document.getElementById('batch-prediction-form');
    if (batchForm) {
        batchForm.addEventListener('submit', async function(e) {
            e.preventDefault(); // Evitar recargar la página
            const form = e.target;
            const formData = new FormData(form);

            // Depurar los datos que se envían al backend
            console.log("Datos enviados al backend:");
            formData.forEach((value, key) => {
                console.log(`${key}: ${value}`);
            });
            
            const resultDiv = document.getElementById('batch-result');
            const graphDiv = document.getElementById('batch-graph');
            
            resultDiv.innerHTML = '<div class="loading">Procesando lote de datos</div>';
            resultDiv.className = 'result-box';
            graphDiv.innerHTML = '';

            try {
                const response = await fetch('/batch_predict', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    // Formatear la matriz de confusión para mejor visualización
                    let confusionMatrixHTML = '';
                    if (data.confusion_matrix) {
                        confusionMatrixHTML = `
                            <div class="confusion-matrix">
                                <h4>Matriz de Confusión:</h4>
                                <pre>${JSON.stringify(data.confusion_matrix, null, 2)}</pre>
                            </div>
                        `;
                    }
                    
                    resultDiv.innerHTML = `
                        <div class="success">
                            <h3><i class="fas fa-check-circle"></i> Análisis por Lote Completado</h3>
                            ${confusionMatrixHTML}
                            <p><strong>Exactitud:</strong> ${data.accuracy}</p>
                        </div>
                    `;
                    
                    if (data.graph) {
                        // Mostrar el gráfico de la matriz de confusión
                        graphDiv.innerHTML = `
                            <h3>Visualización de Resultados</h3>
                            <img src="data:image/png;base64,${data.graph}" alt="Matriz de Confusión">
                        `;
                    }
                } else {
                    const error = await response.json();
                    resultDiv.innerHTML = `
                        <div class="error">
                            <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
                            <p>${error.error || 'Ha ocurrido un error al procesar la solicitud'}</p>
                        </div>
                    `;
                }
            } catch (err) {
                resultDiv.innerHTML = `
                    <div class="error">
                        <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
                        <p>Error al procesar la solicitud: ${err.message}</p>
                    </div>
                `;
            }
        });
    }

    // Validación de formularios
    const allForms = document.querySelectorAll('form');
    allForms.forEach(form => {
        const inputs = form.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            input.addEventListener('invalid', function() {
                this.classList.add('error-input');
            });
            
            input.addEventListener('input', function() {
                this.classList.remove('error-input');
            });
        });
    });
});