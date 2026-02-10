from django.contrib import admin
from .models import Gallery


@admin.register(Gallery)
class GalleryAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "owner", "model_path", "created_at")
    list_filter = ("owner", "created_at")
    search_fields = ("name", "owner__username")
