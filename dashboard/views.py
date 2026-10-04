import json

from django.http import StreamingHttpResponse
from django.shortcuts import render

from model.predict import predict_text
from model.train_dashboard import train_model


# ==========================================
# HOME
# ==========================================

def index(request):

    result = None
    text = ""

    if request.method == "POST":

        text = request.POST.get(
            "text",
            ""
        ).strip()

        if text:

            result = predict_text(text)

            result["confidence_percent"] = (
                result["confidence"] * 100
            )

    return render(
        request,
        "dashboard/index.html",
        {
            "result": result,
            "text": text,
        }
    )


# ==========================================
# SSE TRAINING
# ==========================================

def train_stream(request):

    def event_stream():

        # เริ่มต้น
        yield (
            "data: "
            + json.dumps({
                "type": "start"
            })
            + "\n\n"
        )

        # Training
        for result in train_model():

            if result.get("type") == "final":

                data = {
                    "type": "final",
                    "test_loss": result["test_loss"],
                    "test_accuracy": result["test_accuracy"]
                }
                
                yield (
                    "data: "
                    + json.dumps(data)
                    + "\n\n"
                )

            else:

                data = {
                    "type": "epoch",
                    "epoch": result["epoch"],
                    "loss": result["loss"],
                    "accuracy": result["accuracy"],
                    "val_loss": result["val_loss"],
                    "val_accuracy": result["val_accuracy"]
                }

                yield (
                    "data: "
                    + json.dumps(data)
                    + "\n\n"
                )

        # เสร็จแล้ว
        yield (
            "data: "
            + json.dumps({
                "type": "complete"
            })
            + "\n\n"
        )

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream"
    )

    response["Cache-Control"] = "no-cache"

    return response