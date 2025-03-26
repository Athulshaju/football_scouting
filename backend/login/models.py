from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class ScoutProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    phone_number = models.CharField(max_length=15)
    club_name = models.CharField(max_length=50)
    city = models.CharField(max_length=50)
    country = models.CharField(max_length=50)
    def __str__(self):
        return self.user.username

