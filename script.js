// import { Tensor, Session } from 'onnxjs';

// script.js
// JavaScript code for webcam access and model inference using ONNX.js

// Get the video element
const video = document.getElementById("videoElement");

// Check if the browser supports navigator.mediaDevices.getUserMedia
if (navigator.mediaDevices.getUserMedia) {
    // Get access to the webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            // Assign the video stream to the video element
            video.srcObject = stream;
        })
        .catch(function (error) {
            console.log("Error accessing the webcam: " + error);
        });
}

// Load the ONNX model
const session = InferenceSession.create(
	"./model.onnx",
	{
	  executionProviders: ["webgl"],
	}
  );

// Function to run model inference on a frame
async function runModelOnFrame(frameData) {
    try {
        // Preprocess the frame if necessary
        // Convert the frame data to a tensor
        const tensor = new onnx.Tensor(new Float32Array(frameData.data), 'float32', [1, frameData.height, frameData.width, 3]);

        // Run inference on the tensor
        const outputMap = await session.run([tensor]);
        const outputTensor = outputMap.values().next().value;
        const outputData = outputTensor.data;

        // Process the model output as needed
        // For demonstration purposes, let's just display the raw output data
        console.log("Model output:", outputData);
    } catch (error) {
        console.error("Error running model on frame:", error);
    }
}


// Continuously process video frames
video.addEventListener("play", function () {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    const resizedWidth = 128;
    const resizedHeight = 128;
    canvas.width = resizedWidth;
    canvas.height = resizedHeight;

    setInterval(async function () {
        // Draw the current frame onto the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get the image data from the canvas
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        
        // Run the model on the image data
        await runModelOnFrame(imageData);

    }, 100); // Adjust the interval as needed
});
