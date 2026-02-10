from django.urls import path
from . import views

urlpatterns = [
    # Pages + actions combined
    path("train/", views.train_model, name="train"),
    path("verify/", views.verify_face, name="verify"),

    # Gallery features
    path("galleries/", views.gallery_list_page, name="gallery_list"),
    path("delete-person/", views.delete_person, name="delete_person"),
    path("delete-gallery/<str:gallery_name>/", views.delete_gallery, name="delete_gallery"),
    path("download/<str:gallery_name>/", views.download_gallery, name="download_gallery"),

    
]
