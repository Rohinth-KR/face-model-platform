from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from faceapi import views as face_views

urlpatterns = [
    path("admin/", admin.site.urls),

    # 🔐 AUTH
    path("login/", auth_views.LoginView.as_view(), name="login"),
    path("logout/", face_views.logout_view, name="logout"),

    path("signup/", face_views.signup_view, name="signup"),

    # 🚀 APP
    path("api/", include("faceapi.urls")),
    


]
