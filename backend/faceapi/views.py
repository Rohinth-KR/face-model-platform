import os
import shutil
import numpy as np
import joblib

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login

from face_detector import detect_and_crop_face
from embedding_generator import generate_embeddings

from .models import Gallery


# =========================
# CONFIG
# =========================

BASE_ML_DIR = "ml"
DATA_DIR = os.path.join(BASE_ML_DIR, "data")
MODEL_DIR = os.path.join(BASE_ML_DIR, "models")
THRESHOLD = 0.45


# =========================
# HELPERS (DB FIRST)
# =========================

def get_gallery_persons(user, gallery_name):
    try:
        gallery = Gallery.objects.get(owner=user, name=gallery_name)
    except Gallery.DoesNotExist:
        return []

    if not os.path.exists(gallery.model_path):
        return []

    data = joblib.load(gallery.model_path)
    return list(data.keys())


# =========================
# PAGE VIEWS
# =========================

@login_required
def train_page(request):
    return render(request, "faceapi/train.html")


@login_required
def verify_page(request):
    gallery_name = request.GET.get("gallery_name")
    return render(
        request,
        "faceapi/verify.html",
        {"gallery": gallery_name}
    )


# =========================
# TRAIN MULTI-PERSON
# =========================

@csrf_exempt
@login_required
def train_model(request):
    context = {}

    # ---------- GET ----------
    if request.method == "GET":
        gallery_name = request.GET.get("gallery_name")
        if gallery_name:
            context["gallery"] = gallery_name
            context["persons"] = get_gallery_persons(request.user, gallery_name)
        return render(request, "faceapi/train.html", context)

    # ---------- POST ----------
    gallery_name = request.POST.get("gallery_name")
    person_name = request.POST.get("person_name")
    images = request.FILES.getlist("images")

    if not gallery_name or not person_name or not images:
        context["error"] = "Gallery name, person name, and images are required"
        return render(request, "faceapi/train.html", context)

    user = request.user

    gallery_obj, _ = Gallery.objects.get_or_create(
        owner=user,
        name=gallery_name,
        defaults={
            "model_path": os.path.join(
                MODEL_DIR, f"user_{user.id}", f"{gallery_name}.pkl"
            )
        }
    )

    model_path = gallery_obj.model_path

    base_dir = os.path.join(
        DATA_DIR, f"user_{user.id}", gallery_name, person_name
    )
    raw_dir = os.path.join(base_dir, "raw")
    face_dir = os.path.join(base_dir, "faces")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for img in images:
        with open(os.path.join(raw_dir, img.name), "wb+") as f:
            for chunk in img.chunks():
                f.write(chunk)

    for img_name in os.listdir(raw_dir):
        detect_and_crop_face(
            os.path.join(raw_dir, img_name),
            os.path.join(face_dir, img_name)
        )

    embeddings = generate_embeddings(face_dir)

    if len(embeddings) < 5:
        context["error"] = "Not enough valid face images (min 5 required)"
        return render(request, "faceapi/train.html", context)

    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding /= np.linalg.norm(mean_embedding)

    gallery_data = joblib.load(model_path) if os.path.exists(model_path) else {}
    gallery_data[person_name] = mean_embedding
    joblib.dump(gallery_data, model_path)

    context.update({
        "success": "Person added to gallery successfully",
        "gallery": gallery_name,
        "person": person_name,
        "count": len(gallery_data),
        "persons": list(gallery_data.keys())
    })

    return render(request, "faceapi/train.html", context)


# =========================
# VERIFY FACE
# =========================
@csrf_exempt
@login_required
def verify_face(request):
    context = {}

    if request.method == "POST":
        gallery_name = request.POST.get("gallery_name")
        image = request.FILES.get("image")

        if not gallery_name or not image:
            context["error"] = "Gallery name and image are required"
            return render(request, "faceapi/verify.html", context)

        user = request.user
        model_path = os.path.join(
            MODEL_DIR, f"user_{user.id}", f"{gallery_name}.pkl"
        )

        if not os.path.exists(model_path):
            context["error"] = "Gallery not found"
            return render(request, "faceapi/verify.html", context)

        # Temp folders
        temp_dir = "ml/temp_verify"
        raw_dir = os.path.join(temp_dir, "raw")
        face_dir = os.path.join(temp_dir, "faces")

        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(raw_dir)
        os.makedirs(face_dir)

        img_path = os.path.join(raw_dir, image.name)
        with open(img_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        detect_and_crop_face(
            img_path,
            os.path.join(face_dir, "face.jpg")
        )

        embeddings = generate_embeddings(face_dir)

        if len(embeddings) == 0:
            context["error"] = "No face detected"
            return render(request, "faceapi/verify.html", context)

        test_embedding = embeddings[0]
        gallery = joblib.load(model_path)

        best_person = None
        best_score = -1.0  # Python float

        for person, ref_embedding in gallery.items():
            score = float(np.dot(ref_embedding, test_embedding))  # 🔥 FIX
            if score > best_score:
                best_score = score
                best_person = person

        if best_score >= THRESHOLD:
            context.update({
                "result": "MATCH",
                "person": best_person,
                "similarity": round(best_score, 3)
            })
        else:
            context.update({
                "result": "NO MATCH",
                "similarity": round(best_score, 3)
            })

    return render(request, "faceapi/verify.html", context)


# =========================
# DELETE PERSON
# =========================

@login_required
def delete_person(request):
    if request.method != "POST":
        return redirect("/api/train/")

    gallery_name = request.POST.get("gallery_name")
    person_name = request.POST.get("person_name")

    try:
        gallery = Gallery.objects.get(owner=request.user, name=gallery_name)
    except Gallery.DoesNotExist:
        return redirect("/api/train/")

    if os.path.exists(gallery.model_path):
        data = joblib.load(gallery.model_path)
        data.pop(person_name, None)
        joblib.dump(data, gallery.model_path)

    return redirect(f"/api/train/?gallery_name={gallery_name}")


# =========================
# DELETE GALLERY
# =========================

@login_required
def delete_gallery(request, gallery_name):
    try:
        gallery = Gallery.objects.get(owner=request.user, name=gallery_name)
    except Gallery.DoesNotExist:
        return redirect("/api/train/")

    if os.path.exists(gallery.model_path):
        os.remove(gallery.model_path)

    data_dir = os.path.join(DATA_DIR, f"user_{request.user.id}", gallery_name)
    shutil.rmtree(data_dir, ignore_errors=True)

    gallery.delete()
    return redirect("/api/train/")


# =========================
# SIGNUP
# =========================

def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if not username or not password1 or not password2:
            return render(request, "registration/signup.html", {"error": "All fields required"})

        if password1 != password2:
            return render(request, "registration/signup.html", {"error": "Passwords do not match"})

        if User.objects.filter(username=username).exists():
            return render(request, "registration/signup.html", {"error": "Username already exists"})

        user = User.objects.create_user(username=username, password=password1)
        login(request, user)
        return redirect("/api/train/")

    return render(request, "registration/signup.html")

@login_required
def gallery_list_page(request):
    galleries = Gallery.objects.filter(owner=request.user)

    data = []
    for g in galleries:
        persons = []
        if g.model_path and os.path.exists(g.model_path):
            persons = list(joblib.load(g.model_path).keys())

        data.append({
            "name": g.name,
            "persons": persons
        })

    return render(
        request,
        "faceapi/gallery_list.html",
        {
            "galleries": data
        }
    )

from django.http import FileResponse, Http404

@login_required
def download_gallery(request, gallery_name):
    try:
        gallery = Gallery.objects.get(
            owner=request.user,
            name=gallery_name
        )
    except Gallery.DoesNotExist:
        raise Http404("Gallery not found")

    model_path = gallery.model_path

    if not model_path or not os.path.exists(model_path):
        raise Http404("Model file missing")

    return FileResponse(
        open(model_path, "rb"),
        as_attachment=True,
        filename=f"{gallery_name}.pkl"
    )
from django.contrib.auth import logout
from django.shortcuts import redirect

def logout_view(request):
    logout(request)
    return redirect("/signup/")
