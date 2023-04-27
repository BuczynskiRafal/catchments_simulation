import os
from django.conf import settings

import pandas as pd
import plotly.express as px
from django.shortcuts import render
from django.shortcuts import get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth import get_user_model

from main.forms import ContactForm
from main.forms import UserProfileForm

from . import services
from .forms import ContactForm


def plot(path, x, y="runoff", xaxes=False, start=0, stop=100, title=None, rename_labels=False, x_name=None, y_name=None):
    df = pd.read_excel(path)
    fig = px.line(df, x, y, title=title)
    if xaxes:
        fig.update_xaxes(range=[start, stop])

    if rename_labels:
        fig.update_xaxes(title_text=x_name)  # Dodaj tę linię
        fig.update_yaxes(title_text=y_name)  # Zaktualizuj tę linię
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
        )
    )
    plot_div = fig.to_html(full_html=False)
    return plot_div

def main_view(request):
    context = {
        "plot_slope": plot(os.path.join(settings.BASE_DIR, "data", "df_slope.xlsx"), x="slope", xaxes=True, title="Dependence of runoff on subcatchment slope.", rename_labels=True, x_name="Percent Slope [-]", y_name="Runoff [m3]"),
        "plot_impervious": plot(os.path.join(settings.BASE_DIR, "data", "df_percent_imprevious.xlsx"), x="percent_impervious", xaxes=True, title="Dependence of runoff on subcatchment imprevious.", rename_labels=True, x_name="Imprevious [%]", y_name="Runoff [m3]"),
        "plot_area": plot(os.path.join(settings.BASE_DIR, "data", "df_area.xlsx"), x="area", xaxes=False, title="Dependence of runoff on subcatchment area.", rename_labels=True, x_name="Area [ha]", y_name="Runoff [m3]"),
        "plot_width": plot(os.path.join(settings.BASE_DIR, "data", "df_width.xlsx"), x="width", xaxes=True,  stop=1000, title="Dependence of runoff on subcatchment width.", rename_labels=True, x_name="Width [m]", y_name="Runoff [m3]"),
        "manning_impervious": plot(os.path.join(settings.BASE_DIR, "data", "df_n_impervious.xlsx"), x="N-Imperv", xaxes=False,  title="Dependence of runoff on Manning's impervious.", rename_labels=True, x_name="Manning's N-Imperv [-]", y_name="Runoff [m3]"),
        "manning_pervious": plot(os.path.join(settings.BASE_DIR, "data", "df_n_perv.xlsx"), x="N-Perv", xaxes=False,  title="Dependence of runoff on Manning's pervious.", rename_labels=True, x_name="Manning's N-Perv [-]", y_name="Runoff [m3]"),
        "destore_impervious": plot(os.path.join(settings.BASE_DIR, "data", "df_s_imperv.xlsx"), x="Destore-Imperv", xaxes=False,  title="Dependence of runoff on Destore impervious.", rename_labels=True, x_name="Destore Impervious [inches]", y_name="Runoff [m3]"),
        "destore_pervious": plot(os.path.join(settings.BASE_DIR, "data", "df_s_perv.xlsx"), x="Destore-Perv", xaxes=False,  title="Dependence of runoff on Destore pervious.", rename_labels=True, x_name="Destore Pervious [inches]", y_name="Runoff [m3]"),
        "zero_impervious": plot(os.path.join(settings.BASE_DIR, "data", "df_zero_imperv.xlsx"), x="Zero-Imperv", xaxes=False,  title="Dependence of runoff on zero impervious area.", rename_labels=True, x_name="Zero Impervious [-]", y_name="Runoff [m3]"),
        }
    return render(request, "main/main_view.html", context)

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

def simulation(request):
    return render(request, 'main/simulation.html')


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
