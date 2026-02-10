from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.shortcuts import redirect
from faceapi import views as face_views


def home(request):
    # Redirect root URL to login page
    return redirect("login")


urlpatterns = [
    # 🌍 ROOT FIX
    path("", home, name="home"),

    # 🔐 AUTH
    path("login/", auth_views.LoginView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("signup/", face_views.signup_view, name="signup"),

    # 🚀 APP
    path("api/", include("faceapi.urls")),

    path("admin/", admin.site.urls),
]
