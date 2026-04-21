/** @format */

window.addEventListener("load", () => {
	const canvas = document.querySelector("#myCanvas");
	const ctx = canvas.getContext("2d");

	const previewCanvas = document.querySelector("#previewCanvas");
	const previewCtx = previewCanvas ? previewCanvas.getContext("2d") : null;
	const predictionResult = document.querySelector("#predictionResult");
	const INK_THRESHOLD = 30;
	const CROP_MARGIN = 20;

	function fillBlack() {
		ctx.save();
		ctx.fillStyle = "black";
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.restore();
	}

	fillBlack();

	let painting = false;

	function getMousePos(e) {
		const rect = canvas.getBoundingClientRect();
		const scaleX = canvas.width / rect.width;
		const scaleY = canvas.height / rect.height;

		return {
			x: (e.clientX - rect.left) * scaleX,
			y: (e.clientY - rect.top) * scaleY,
		};
	}

	function startPosition(e) {
		painting = true;
		draw(e);
	}

	function finishedPosition() {
		painting = false;
		ctx.beginPath();
	}

	function draw(e) {
		if (!painting) return;

		const pos = getMousePos(e);
		ctx.lineWidth = 40;
		ctx.lineCap = "round";
		ctx.strokeStyle = "white";

		ctx.lineTo(pos.x, pos.y);
		ctx.stroke();
		ctx.beginPath();
		ctx.moveTo(pos.x, pos.y);
	}

	canvas.addEventListener("mousedown", startPosition);
	canvas.addEventListener("mouseup", finishedPosition);
	canvas.addEventListener("mousemove", draw);

	function getGrayscaleData() {
		const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		const grayscaleData = new Uint8ClampedArray(
			imageData.width * imageData.height,
		);

		for (let i = 0; i < imageData.data.length; i += 4) {
			const r = imageData.data[i];
			const g = imageData.data[i + 1];
			const b = imageData.data[i + 2];
			grayscaleData[i / 4] = (r + g + b) / 3;
		}

		return { grayscaleData, width: imageData.width, height: imageData.height };
	}

	function getInkMask(grayscaleData, threshold) {
		const mask = new Uint8Array(grayscaleData.length);

		for (let i = 0; i < grayscaleData.length; i += 1) {
			mask[i] = grayscaleData[i] > threshold ? 1 : 0;
		}

		return mask;
	}

	function getInkBoundingBox(mask, width, height) {
		let minX = width;
		let minY = height;
		let maxX = -1;
		let maxY = -1;

		for (let y = 0; y < height; y += 1) {
			for (let x = 0; x < width; x += 1) {
				if (!mask[y * width + x]) continue;

				if (x < minX) minX = x;
				if (x > maxX) maxX = x;
				if (y < minY) minY = y;
				if (y > maxY) maxY = y;
			}
		}

		if (maxX === -1) return null;

		return { minX, minY, maxX, maxY };
	}

	function createSquareCropCanvas(bounds, margin) {
		const cropMinX = Math.max(0, bounds.minX - margin);
		const cropMinY = Math.max(0, bounds.minY - margin);
		const cropMaxX = Math.min(canvas.width - 1, bounds.maxX + margin);
		const cropMaxY = Math.min(canvas.height - 1, bounds.maxY + margin);

		const cropWidth = cropMaxX - cropMinX + 1;
		const cropHeight = cropMaxY - cropMinY + 1;
		const side = Math.max(cropWidth, cropHeight);

		const normalized = document.createElement("canvas");
		normalized.width = side;
		normalized.height = side;

		const nctx = normalized.getContext("2d");
		nctx.fillStyle = "black";
		nctx.fillRect(0, 0, side, side);

		const dx = Math.floor((side - cropWidth) / 2);
		const dy = Math.floor((side - cropHeight) / 2);

		nctx.drawImage(
			canvas,
			cropMinX,
			cropMinY,
			cropWidth,
			cropHeight,
			dx,
			dy,
			cropWidth,
			cropHeight,
		);

		return normalized;
	}

	function createMnist28x28Canvas(normalizedCanvas) {
		const mnistCanvas = document.createElement("canvas");
		mnistCanvas.width = 28;
		mnistCanvas.height = 28;

		const mctx = mnistCanvas.getContext("2d");
		mctx.fillStyle = "black";
		mctx.fillRect(0, 0, 28, 28);
		mctx.imageSmoothingEnabled = true;
		mctx.imageSmoothingQuality = "high";
		mctx.drawImage(
			normalizedCanvas,
			0,
			0,
			normalizedCanvas.width,
			normalizedCanvas.height,
			4,
			4,
			20,
			20,
		);

		return mnistCanvas;
	}

	function centerOfMassAlign28x28(sourceCanvas) {
		const sctx = sourceCanvas.getContext("2d");
		const imageData = sctx.getImageData(0, 0, 28, 28);
		const data = imageData.data;

		let sumI = 0;
		let sumX = 0;
		let sumY = 0;

		for (let y = 0; y < 28; y += 1) {
			for (let x = 0; x < 28; x += 1) {
				const idx = (y * 28 + x) * 4;
				const r = data[idx];
				const g = data[idx + 1];
				const b = data[idx + 2];
				const intensity = (r + g + b) / 3;

				sumI += intensity;
				sumX += x * intensity;
				sumY += y * intensity;
			}
		}

		if (sumI === 0) return sourceCanvas;

		const cx = sumX / sumI;
		const cy = sumY / sumI;
		const dx = Math.round(14 - cx);
		const dy = Math.round(14 - cy);

		const aligned = document.createElement("canvas");
		aligned.width = 28;
		aligned.height = 28;

		const actx = aligned.getContext("2d");
		actx.fillStyle = "black";
		actx.fillRect(0, 0, 28, 28);
		actx.imageSmoothingEnabled = false;
		actx.drawImage(sourceCanvas, dx, dy);

		return aligned;
	}

	function extract784(sourceCanvas) {
		const sctx = sourceCanvas.getContext("2d");
		const imageData = sctx.getImageData(0, 0, 28, 28);
		const data = imageData.data;
		const pixels = new Uint8ClampedArray(28 * 28);

		for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
			const r = data[i];
			const g = data[i + 1];
			const b = data[i + 2];
			pixels[p] = Math.round((r + g + b) / 3);
		}

		return pixels;
	}

	function drawPreview(sourceCanvas) {
		if (!previewCanvas || !previewCtx) return;

		previewCtx.imageSmoothingEnabled = false;
		previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
		previewCtx.drawImage(
			sourceCanvas,
			0,
			0,
			previewCanvas.width,
			previewCanvas.height,
		);
	}

	function clearCanvas() {
		fillBlack();
		if (previewCanvas && previewCtx) {
			previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
		}
		if (predictionResult) {
			predictionResult.textContent = "Prediction: --";
		}
	}

	async function predict() {
		const { grayscaleData, width, height } = getGrayscaleData();
		const inkMask = getInkMask(grayscaleData, INK_THRESHOLD);
		const bounds = getInkBoundingBox(inkMask, width, height);

		if (!bounds) {
			console.log("Blank canvas: no ink detected.");
			if (previewCanvas && previewCtx) {
				previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
			}
			if (predictionResult) {
				predictionResult.textContent = "Prediction: blank canvas";
			}
			return;
		}

		const normalizedCanvas = createSquareCropCanvas(bounds, CROP_MARGIN);
		const final28 = createMnist28x28Canvas(normalizedCanvas);
		const aligned28 = centerOfMassAlign28x28(final28);
		const pixels784 = extract784(aligned28);

		drawPreview(aligned28);

		let min = 255;
		let max = 0;
		for (let i = 0; i < pixels784.length; i += 1) {
			const value = pixels784[i];
			if (value < min) min = value;
			if (value > max) max = value;
		}

		console.log("pixels length:", pixels784.length);
		console.log("pixels[0..19]:", Array.from(pixels784.slice(0, 20)));
		console.log("pixel stats:", { min, max });

		if (predictionResult) {
			predictionResult.textContent = "Prediction: running...";
		}

		try {
			const response = await fetch("http://127.0.0.1:8000/predict", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ pixels: Array.from(pixels784) }),
			});

			const data = await response.json();
			if (!response.ok) {
				throw new Error(data.error || "Prediction request failed");
			}

			console.log("prediction result:", data);
			if (predictionResult) {
				const confidencePct = (Number(data.confidence) * 100).toFixed(1);
				predictionResult.textContent = `Prediction: ${data.digit} (${confidencePct}%)`;
			}
		} catch (error) {
			console.error("predict error:", error);
			if (predictionResult) {
				predictionResult.textContent = "Prediction error: backend unavailable";
			}
		}
	}

	const clearButton = document.querySelector("#clearButton");
	clearButton.addEventListener("click", clearCanvas);

	const predictButton = document.querySelector("#predictButton");
	predictButton.addEventListener("click", predict);
});
