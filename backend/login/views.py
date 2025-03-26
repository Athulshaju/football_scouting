from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
# Create your views here.

def register(request):
    if request.method == 'POST':
        name=request.POST['name']
        username=request.POST['username']
        password1=request.POST['password']
        password2=request.POST['password2']
        email=request.POST['email']