using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Python.Runtime;
using System.Text;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

InitPython();
InitPyModelModule();

app.MapPost("/predict", async (HttpRequest request) =>
{
    var saveDir = "uploads";
    Directory.CreateDirectory(saveDir);

    var file = request.Form.Files[0];
    var path = Path.Combine(saveDir, file.FileName);

    using (var fs = File.Create(path))
        await file.CopyToAsync(fs);

    string result;
    double conf;

    using (Py.GIL())
    {
        dynamic mod = Py.Import("captcha_bridge");
        dynamic output = mod.predict_one(path);
        result = output[0].ToString();
        conf = (double)output[1];
    }

    return Results.Ok(new { text = result, confidence = conf });
});

app.Run();


// ---------------- PYTHON INIT ----------------

static void InitPython()
{
    Console.WriteLine("InitPython start");

    if (OperatingSystem.IsMacOS())
    {
        string pyHome =
        "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11";

        string stdLib = pyHome + "/lib/python3.11";

        string venvSite =
        "/Volumes/samsung_980/projects/captcha-reader/env/lib/python3.11/site-packages";

        string pyDll = pyHome + "/lib/libpython3.11.dylib";

        Runtime.PythonDLL = pyDll;

        // ✅ REQUIRED for mac framework python
        Environment.SetEnvironmentVariable("PYTHONHOME", pyHome);
        Environment.SetEnvironmentVariable("PYTHONPATH", stdLib + ":" + venvSite);

        PythonEngine.PythonHome = pyHome;
        PythonEngine.PythonPath = stdLib + ":" + venvSite;

        Console.WriteLine("PythonDLL = " + pyDll);
        Console.WriteLine("PYTHONHOME = " + pyHome);
        Console.WriteLine("PYTHONPATH = " + stdLib + ":" + venvSite);
    }

    Console.WriteLine("Calling PythonEngine.Initialize...");
    PythonEngine.Initialize();
    Console.WriteLine("PythonEngine.Initialize OK");

    using (Py.GIL())
    {
        Console.WriteLine("GIL acquired");
        dynamic sys = Py.Import("sys");
        Console.WriteLine("sys imported");
        Console.WriteLine(sys.version);
    }
}

// ---------------- PYTHON MODULE ----------------

static void InitPyModelModule()
{
    string py = @"
import cv2
import numpy as np
import string
import tensorflow as tf
from keras import models

IMG_W = 200
IMG_H = 50

TTA_RUNS = 12
TTA_RUNS_HARD = 24
CONF_THRESHOLD = 0.82

characters = string.ascii_letters + string.digits + '@=#'
idx_to_char = {i: c for i, c in enumerate(characters)}
blank_index = len(characters)

class CTCModel(tf.keras.Model):
    pass

MODEL_PATH = 'captcha_ctc_adapted_v3.keras'

model = models.load_model(
    MODEL_PATH,
    custom_objects={'CTCModel': CTCModel},
    compile=False
)

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.equalizeHist(img)
    img = img.astype('float32') / 255.0
    return img

def tta_variant(img):
    v = img.copy()
    if np.random.rand() < 0.5:
        v = cv2.dilate(v, np.ones((2,2),np.uint8))
    else:
        v = cv2.erode(v, np.ones((2,2),np.uint8))

    alpha = np.random.uniform(0.9, 1.15)
    beta  = np.random.uniform(-0.05, 0.05)
    v = np.clip(v * alpha + beta, 0, 1)

    if np.random.rand() < 0.5:
        c = np.random.randint(2,10)
        v = v[:, c:-c]
        v = cv2.resize(v, (IMG_W, IMG_H))
    return v

def decode_with_conf(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded,_ = tf.keras.backend.ctc_decode(pred,input_len,greedy=True)
    seqs = decoded[0].numpy()

    texts=[]; confs=[]
    for b, seq in enumerate(seqs):
        chars=[]; probs=[]
        for t, idx in enumerate(seq):
            if idx == -1 or idx == blank_index:
                continue
            if idx in idx_to_char:
                chars.append(idx_to_char[idx])
                probs.append(np.max(pred[b,t]))
        conf = float(np.mean(probs)) if probs else 0.0
        texts.append(''.join(chars))
        confs.append(conf)
    return texts, confs

def tta_predict(img, runs):
    variants=[tta_variant(img) for _ in range(runs)]
    batch=np.array(variants)[...,None]
    pred=model.predict(batch,verbose=0)

    texts, confs = decode_with_conf(pred)
    best=max(set(texts), key=texts.count)
    vote_conf=texts.count(best)/runs
    mean_conf=np.mean([c for t,c in zip(texts,confs) if t==best])
    final_conf=0.6*vote_conf + 0.4*mean_conf
    return best, final_conf

def predict_one(path):
    img=preprocess(path)
    pred, conf = tta_predict(img, TTA_RUNS)
    if conf < CONF_THRESHOLD:
        pred, conf = tta_predict(img, TTA_RUNS_HARD)
    return pred, float(conf)
";

    using (Py.GIL())
    {
        Console.WriteLine("Loading Python ML module...");
        PythonEngine.Exec(py);
        Console.WriteLine("Python ML module loaded");
    }
}
