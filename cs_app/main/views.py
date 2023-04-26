from django.shortcuts import render
from django.shortcuts import get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth import get_user_model

from main.forms import ContactForm
from main.forms import UserProfileForm

from . import services
from .forms import ContactForm


def main_view(request):
    return render(request, 'main/main_view.html')


def about(request):
    return render(request, 'main/about.html')


def contact(request):
    if request.method == "POST":
        form = ContactForm(data=request.POST)
        if form.is_valid():
            services.send_message(form.cleaned_data)
            return HttpResponseRedirect(reverse('contact'))
    else:
        form = ContactForm()
    return render(request, 'main/contact.html', {'form': form})


def user_profile(request, user_id):
    user = get_object_or_404(get_user_model(), id=user_id)
    if request.method == "POST":
        try:
            profile = user.userprofile
            form = UserProfileForm(request.POST, instance=profile)
        except AttributeError:
            pass
        if form.is_valid():
            form.save()
    else:
        try:
            profile = user.userprofile
            form = UserProfileForm(instance=profile)
        except AttributeError:
            form = UserProfileForm(initial={"user":user, "bio": ""})
        if request.user != user:
            for field in form.fields:
                form.fields[field].disabled = True
            form.helper.inputs = []
    return render(request, 'main/userprofile.html', {'form': form})