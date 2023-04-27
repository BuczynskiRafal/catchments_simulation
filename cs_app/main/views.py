import os
import pandas as pd
import plotly.express as px
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, FileResponse, JsonResponse
from django.urls import reverse
from django.contrib.auth import get_user_model

from main.forms import ContactForm
from main.forms import UserProfileForm

from . import services
from .forms import ContactForm, SimulationForm

from catchment_simulation.catchment_features_simulation import FeaturesSimulation


def plot(x, y="runoff", path=None, df=None, xaxes=False, start=0, stop=100, title=None, rename_labels=False, x_name=None, y_name=None):
    if path is not None:
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

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        filename, file_extension = os.path.splitext(uploaded_file.name)

        if file_extension.lower() == '.inp':
            file_path = os.path.join('uploaded_files', filename + file_extension)
            print(f"file_path:: {file_path}")
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Przechowuj ścieżkę do pliku w sesji
            request.session['uploaded_file_path'] = file_path

            return JsonResponse({'message': 'File was sent.'})
        else:
            return JsonResponse({'error': 'Invalid file type. Please upload a .inp file.'})
    return JsonResponse({'error': 'Error occurred while sending file.'})

def simulation_view(request):
    show_download_button = False
    user_plot = None
    if request.method == 'POST':
        form = SimulationForm(request.POST)
        if form.is_valid():
            method_name = form.cleaned_data['option']
            start = form.cleaned_data['start']
            stop = form.cleaned_data['stop']
            step = form.cleaned_data['step']
            catchment_name = form.cleaned_data['catchment_name']

            # Pobierz ścieżkę do pliku z sesji
            uploaded_file_path = request.session.get('uploaded_file_path', os.path.abspath('catchment_simulation/example.inp'))

            model = FeaturesSimulation(subcatchment_id=catchment_name, raw_file=uploaded_file_path)

            method = getattr(model, method_name)
            df = method(start=start, stop=stop, step=step)

            fetaure = {
                'simulate_percent_slope': "PercSlope",
                'simulate_area': "Area",
                'simulate_width': "Width",
                'simulate_percent_impervious': "PercImperv",
                'simulate_percent_zero_imperv': "Zero-Imperv",
            }

            show_download_button = True

            output_file_path = 'output_files/simulation_result.xlsx'
            df.to_excel(output_file_path, index=False)

            user_plot = plot(df=df, x=fetaure[method_name], xaxes=False, title=f"Dependence of runoff on subcatchment {fetaure[method_name]}.")
            
        
            # Przekieruj do innego widoku lub zaktualizuj stronę z wynikami symulacji
            return redirect('main:simulation')  # Przekieruj do widoku 'simulation'

    else:
        form = SimulationForm()

    return render(request, 'main/simulation.html', {'form': form, 'show_download_button': show_download_button, "user_plot": user_plot})




def download_result(request):
    output_file_path = 'output_files/simulation_result.xlsx'

    if os.path.exists(output_file_path):
        response = FileResponse(open(output_file_path, 'rb'), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=simulation_result.xlsx'
        return response
    else:
        return HttpResponseNotFound("File not found.")
