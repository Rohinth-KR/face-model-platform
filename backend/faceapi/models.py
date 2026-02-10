from django.db import models
from django.contrib.auth.models import User


class Gallery(models.Model):
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="galleries"
    )
    name = models.CharField(max_length=100)
    model_path = models.CharField(max_length=255)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("owner", "name")
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.owner.username} → {self.name}"
