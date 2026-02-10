using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Text.Json.Serialization;

const int IMG_W = 200;
const int IMG_H = 50;
const int TTA_RUNS = 6;
const int TTA_RUNS_HARD = 12;
const double CONF_THRESHOLD = 0.82;

string ENV_SAVE_PATH = "CAPTCHA_SAVE_PATH";

string chars =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=#";

var idxToChar = chars.Select((c,i)=>(c,i))
                     .ToDictionary(x=>x.i,x=>x.c);

int blankIndex = chars.Length;

string GetSaveDir()
{
    var env = Environment.GetEnvironmentVariable(ENV_SAVE_PATH);
    if (!string.IsNullOrEmpty(env)) return env;

    var home = Environment.GetFolderPath(
        Environment.SpecialFolder.UserProfile);

    return OperatingSystem.IsWindows()
        ? Path.Combine(home,"Documents","captchas")
        : Path.Combine(home,"captchas");
}

var SAVE_DIR = GetSaveDir();
Directory.CreateDirectory(SAVE_DIR);

Console.WriteLine("Save dir: " + SAVE_DIR);

var session = new InferenceSession("model.onnx");
var inputName = session.InputMetadata.Keys.First();

Console.WriteLine("ONNX input: " + inputName);

float[] Preprocess(string path)
{
    var img = Cv2.ImRead(path, ImreadModes.Grayscale);
    Cv2.Resize(img, img, new Size(IMG_W, IMG_H));
    Cv2.EqualizeHist(img, img);
    img.ConvertTo(img, MatType.CV_32F, 1.0/255);

    return img.Reshape(1, IMG_W*IMG_H)
              .ToArray<float>();
}

float[] TtaVariant(float[] baseImg)
{
    var mat = new Mat(IMG_H, IMG_W, MatType.CV_32F, baseImg);

    if (Random.Shared.NextDouble()<0.5)
        Cv2.Dilate(mat, mat, Cv2.GetStructuringElement(
            MorphShapes.Rect,new Size(2,2)));
    else
        Cv2.Erode(mat, mat, Cv2.GetStructuringElement(
            MorphShapes.Rect,new Size(2,2)));

    return mat.ToArray<float>();
}

(string,double) Decode(float[] logits, int T, int C)
{
    var text = new List<char>();
    var probs = new List<float>();
    int prev=-1;

    for(int t=0;t<T;t++)
    {
        int best=0;
        float max=0;

        for(int c=0;c<C;c++)
        {
            float v = logits[t*C+c];
            if(v>max){max=v;best=c;}
        }

        if(best==prev || best==blankIndex)
        {
            prev=best;
            continue;
        }

        if(idxToChar.ContainsKey(best))
        {
            text.Add(idxToChar[best]);
            probs.Add(max);
        }

        prev=best;
    }

    return (new string(text.ToArray()),
            probs.Count==0?0:probs.Average());
}

(string,double) RunOnce(float[] img)
{
    var tensor = new DenseTensor<float>(
        img, new[] {1,IMG_H,IMG_W,1});

    var inputs = new List<NamedOnnxValue>{
        NamedOnnxValue.CreateFromTensor(inputName,tensor)
    };

    using var results = session.Run(inputs);
    var data = results.First().AsTensor<float>().ToArray();

    int C = idxToChar.Count + 1;
    int T = data.Length / C;

    return Decode(data,T,C);
}

(string,double) TtaPredict(float[] img,int runs)
{
    var texts=new List<string>();
    var confs=new List<double>();

    for(int i=0;i<runs;i++)
    {
        var v=TtaVariant(img);
        var (t,c)=RunOnce(v);
        texts.Add(t);
        confs.Add(c);
    }

    var best=texts.GroupBy(x=>x)
                  .OrderByDescending(g=>g.Count())
                  .First().Key;

    double vote = texts.Count(x=>x==best)/(double)runs;
    double mean = confs.Where((c,i)=>texts[i]==best).Average();

    return (best,0.6*vote+0.4*mean);
}

var builder = WebApplication.CreateBuilder();
var app = builder.Build();

app.MapPost("/predict_base64", async (CaptchaRequest req)=>
{
    var b64=req.Base64;
    int comma=b64.IndexOf(',');
    if(comma>0) b64=b64[(comma+1)..];

    var bytes=Convert.FromBase64String(b64);

    var name=req.FileName??Guid.NewGuid()+".png";
    var path=Path.Combine(SAVE_DIR,name);

    await File.WriteAllBytesAsync(path,bytes);

    var img=Preprocess(path);

    var (p,c)=TtaPredict(img,TTA_RUNS);
    if(c<CONF_THRESHOLD)
        (p,c)=TtaPredict(img,TTA_RUNS_HARD);

    return Results.Ok(new{
        prediction=p,
        confidence=Math.Round(c,3),
        saved_to=path
    });
});

app.Run("http://0.0.0.0:8080");

public record CaptchaRequest(
 [property:JsonPropertyName("base64")]string Base64,
 [property:JsonPropertyName("file_name")]string? FileName);
