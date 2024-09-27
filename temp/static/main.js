// // JavaScript code to handle camera and image capture
// const video = document.getElementById('cameraFeed');
// const captureButton = document.getElementById('captureButton');
// const capturedImage = document.getElementById('capturedImage');
// const predictButton = document.getElementById('predictButton');
// const imageDataInput = document.getElementById('imageData');

// // Use constraints to define camera properties (adjust as needed)
// const constraints = {
//     video: {
//         width: { ideal: 640 },
//         height: { ideal: 480 },
//     },
// };

// // Access the camera and display the feed
// navigator.mediaDevices.getUserMedia(constraints)
//     .then((stream) => {
//         video.srcObject = stream;
//     })
//     .catch((err) => {
//         console.error('Error accessing camera:', err);
//     });

// // Capture image from the camera when the button is clicked
// captureButton.addEventListener('click', () => {
//     const canvas = document.createElement('canvas');
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     const context = canvas.getContext('2d');
//     context.drawImage(video, 0, 0, canvas.width, canvas.height);

//     // Display the captured image
//     capturedImage.src = canvas.toDataURL('image/png');
//     capturedImage.style.display = 'block';

//     // Enable the predict button
//     predictButton.style.display = 'block';

//     // Convert the captured image to base64 data and set it in the hidden input
//     const imageData = canvas.toDataURL('image/png').replace(/^data:image\/(png|jpg);base64,/, '');
//     imageDataInput.value = imageData;
// });
