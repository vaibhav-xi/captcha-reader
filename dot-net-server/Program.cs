using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
using System.Linq;
using System.Collections.Generic;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using OpenCvSharp;

namespace dot_net_server;

public class Program
{

    static readonly string Chars =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=#";

    const int W = 200;
    const int H = 50;

    const int TTA_RUNS = 12;
    const int TTA_RUNS_HARD = 24;
    const float CONF_THRESHOLD = 0.82f;

    static Random Rng = new();

    public static void Main(string[] args)
    {
        Console.WriteLine(OpenCvSharp.Cv2.GetVersionString());

        var opt = new Microsoft.ML.OnnxRuntime.SessionOptions();
        opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opt.IntraOpNumThreads = Environment.ProcessorCount;

        var session = new InferenceSession("captcha_ctc.onnx", opt);

        var builder = WebApplication.CreateBuilder(args);
        var app = builder.Build();

        Console.WriteLine("OpenCV version: " + Cv2.GetVersionString());

        app.MapPost("/api/captcha", (CaptchaRequest req) =>
        {
            var sw = Stopwatch.StartNew();

            // ---------- base64 decode ----------
            var b64 = req.Base64;
            int comma = b64.IndexOf(',');
            if (comma > 0) b64 = b64[(comma + 1)..];

            byte[] bytes = Convert.FromBase64String(b64);

            using var baseMat = PreprocessBase(bytes);

            var (text, conf) = TtaPredict(session, baseMat, TTA_RUNS);

            if (conf < CONF_THRESHOLD)
                (text, conf) = TtaPredict(session, baseMat, TTA_RUNS_HARD);

            sw.Stop();

            return Results.Ok(new { text, conf, ms = sw.ElapsedMilliseconds });
        });

        app.Run("http://127.0.0.1:5077");
    }

    static Mat PreprocessBase(byte[] bytes)
    {
        var src = Cv2.ImDecode(bytes, ImreadModes.Grayscale);

        var resized = new Mat();
        Cv2.Resize(src, resized, new Size(W, H));      
        Cv2.EqualizeHist(resized, resized);

        src.Dispose();
        return resized;
    }

    // ---------- TTA ----------

    static Mat TtaVariant(Mat src)
    {
        var v = src.Clone();

        if (Rng.NextDouble() < 0.5)
            Cv2.Dilate(v, v, Cv2.GetStructuringElement(
                MorphShapes.Rect, new Size(2,2)));
        else
            Cv2.Erode(v, v, Cv2.GetStructuringElement(
                MorphShapes.Rect, new Size(2,2)));

        double alpha = Rand(0.9, 1.15);
        double beta  = Rand(-0.05, 0.05) * 255.0;

        v.ConvertTo(v, MatType.CV_8U, alpha, beta);

        if (Rng.NextDouble() < 0.5)
        {
            int c = Rng.Next(2, 10);
            var roi = new Rect(c, 0, v.Width - 2*c, v.Height);
            using var cropped = new Mat(v, roi);
            Cv2.Resize(cropped, v, new Size(W, H));
        }

        return v;
    }

    static (string text, float conf) TtaPredict(
        InferenceSession session,
        Mat baseImg,
        int runs)
    {
        var texts = new List<string>();
        var confs = new List<float>();

        for (int i = 0; i < runs; i++)
        {
            using var v = TtaVariant(baseImg);
            var (t,c) = RunOnce(session, v);
            texts.Add(t);
            confs.Add(c);
        }

        var best = texts.GroupBy(x=>x)
                        .OrderByDescending(g=>g.Count())
                        .First().Key;

        float voteConf = texts.Count(x=>x==best)/(float)runs;

        float meanConf = confs.Zip(texts)
                              .Where(p=>p.Second==best)
                              .Select(p=>p.First)
                              .DefaultIfEmpty(0)
                              .Average();

        float finalConf = 0.6f*voteConf + 0.4f*meanConf;

        return (best, finalConf);
    }

    // ---------- Single inference ----------

    static (string text, float conf) RunOnce(
        InferenceSession session,
        Mat img)
    {
        float[] data = new float[H*W];
        int k = 0;

        for (int y=0;y<H;y++)
        for (int x=0;x<W;x++)
            data[k++] = img.At<byte>(y,x)/255f;

        var tensor = new DenseTensor<float>(
            data, new[]{1,H,W,1});

        using var res = session.Run(
            new[]{ NamedOnnxValue.CreateFromTensor("image", tensor) });

        var t = res.First().AsTensor<float>();

        return DecodeWithConf(t);
    }

    // ---------- CTC decode ----------

    static (string,float) DecodeWithConf(Tensor<float> t)
    {
        int time = t.Dimensions[1];
        int classes = t.Dimensions[2];
        int blank = Chars.Length;

        List<int> seq = new();
        List<float> probs = new();

        int prev = -1;

        for(int i=0;i<time;i++)
        {
            int best=0;
            float bestVal=float.MinValue;

            for(int c=0;c<classes;c++)
            {
                float v=t[0,i,c];
                if(v>bestVal){bestVal=v;best=c;}
            }

            if(best!=blank && best!=prev && best<Chars.Length)
            {
                seq.Add(best);
                probs.Add(bestVal);
            }

            prev=best;
        }

        string text = new string(seq.Select(i=>Chars[i]).ToArray());
        float conf = probs.Count>0 ? probs.Average() : 0;

        return (text, conf);
    }

    static double Rand(double a,double b)
        => a + Rng.NextDouble()*(b-a);
}

public record CaptchaRequest(
    [property: JsonPropertyName("base64")] string Base64,
    [property: JsonPropertyName("fileName")] string? FileName);
