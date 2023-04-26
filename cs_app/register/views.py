from django.shortcuts import render
from django.shortcuts import redirect
from django.urls import reverse
from django.contrib.auth.models import Permission

from .forms import RegisterForm


def register(response):
    if response.method == 'POST':
        form = RegisterForm(response.POST)
        if form.is_valid():
            user = form.save()
            permission = Permission.objects.get(name="Can add tag")
            user.user_permissions.add(permission)
        return redirect(reverse('home'))
    else:
        form = RegisterForm()
    return render(response, "account/register.html", {'form': form})