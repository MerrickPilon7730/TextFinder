using System;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Tesseract;
using System.Drawing;
using System.Drawing.Imaging;

class Program
{
	static void Main()
	{
		string videoPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "video.mp4");
		string outputFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "output.text");

		using var ocrEngine = new TesseractEngine(@"C:\Program Files\Tesseract-OCR\tessdata", "eng", EngineMode.LstmOnly);

		ocrEngine.SetVariable("tessedit_char_whitelist","ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}[]();.,+-=*/<>!_\"'#");
		ocrEngine.DefaultPageSegMode = PageSegMode.AutoOsd;


		using var capture = new VideoCapture(videoPath);

		if (!capture.IsOpened)
		{
			Console.WriteLine("Error: Could not open video.");
			return;
		}

		int frameCount = 0;
		int frameRate = (int)capture.Get(CapProp.Fps);
		int frameSkip = frameRate * 1; 

		using StreamWriter writer = new(outputFilePath, false);

		Mat frame = new();

		while (capture.Read(frame))
		{
			frameCount++;

			if (frameCount % frameSkip != 0)
				continue;

			Console.WriteLine($"Processing frame {frameCount}...");

			// Convert frame to grayscale
			Mat grayFrame = new();
			CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

			Mat thresholdFrame = new();
			CvInvoke.Threshold(grayFrame, thresholdFrame, 120, 255, ThresholdType.Binary);


			// Convert Mat to Bitmap
			using Bitmap bmp = MatToBitmap(thresholdFrame);

			string debugPath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "pics"));
			string imagePath = Path.Combine(debugPath, $"frame_{frameCount}.png");
			bmp.Save($"{imagePath}frame_{frameCount}.png", System.Drawing.Imaging.ImageFormat.Png);


			// Convert Bitmap to Pix for Tesseract
			using Pix pix = ConvertBitmapToPix(bmp);

			// Process with OCR
			using var page = ocrEngine.Process(pix);
			Console.WriteLine($"OCR Confidence: {page.GetMeanConfidence() * 100}%");

			string extractedText = page.GetText().Trim();
			if (!string.IsNullOrEmpty(extractedText))
			{
				writer.WriteLine($"Frame {frameCount}: {extractedText}");
				writer.Flush();
			}
		}

		Console.WriteLine($"Processing complete. OCR output saved to {outputFilePath}");
	}

	static Bitmap MatToBitmap(Mat mat)
	{
		Bitmap bitmap = new(mat.Width, mat.Height, PixelFormat.Format8bppIndexed);
		BitmapData data = bitmap.LockBits(
			new Rectangle(0, 0, bitmap.Width, bitmap.Height),
			ImageLockMode.WriteOnly,
			PixelFormat.Format8bppIndexed);

		mat.CopyTo(new Mat(mat.Rows, mat.Cols, DepthType.Cv8U, 1, data.Scan0, mat.Step));
		bitmap.UnlockBits(data);

		return bitmap;
	}

	static Pix ConvertBitmapToPix(Bitmap bmp)
	{
		using MemoryStream ms = new();
		bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
		ms.Position = 0;
		return Pix.LoadFromMemory(ms.ToArray());
	}
}
