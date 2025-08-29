window.addEventListener('load', function () {
    const resultDiv = document.getElementById('result');
    const imageUpload = document.getElementById('imageUpload');
    const predictBtn = document.getElementById('predictBtn');

    resultDiv.innerHTML = '<p>Upload an image of a digit (0-9) and click Predict to get started!</p>';

    predictBtn.addEventListener('click', function () {
        const file = imageUpload.files[0];
        if (!file) {
            resultDiv.innerHTML = '<p><strong>Error:</strong> No file selected</p>';
            return;
        }
        const formData = new FormData();
        formData.append('image', file);

        console.log("Sending file upload to /predict");
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log("Response status:", response.status);
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log("Prediction received:", data);
            resultDiv.innerHTML = `
                <p><strong>Prediction:</strong> ${data.digit}</p>
                <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>
                <p><strong>Probabilities (0-9):</strong> ${data.probabilities.map((p, i) => `${i}: ${(p*100).toFixed(2)}%`).join(', ')}</p>
                ${data.message ? `<p><strong>Message:</strong> ${data.message}</p>` : ''}
            `;
        })
        .catch(err => {
            console.error("Fetch error:", err);
            resultDiv.innerHTML = `<p><strong>Error:</strong> ${err.message}</p>`;
        });
    });
});