import os
import datetime
import pandas as pd
import plotly.express as px
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, FileResponse, JsonResponse, HttpResponse, HttpResponseNotFound
from django.core.files.storage import default_storage
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required

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
        "plot_slope": plot(path=os.path.join(settings.BASE_DIR, "data", "df_slope.xlsx"), x="slope", xaxes=True, title="Dependence of runoff on subcatchment slope.", rename_labels=True, x_name="Percent Slope [-]", y_name="Runoff [m3]"),
        "plot_impervious": plot(path=os.path.join(settings.BASE_DIR, "data", "df_percent_imprevious.xlsx"), x="percent_impervious", xaxes=True, title="Dependence of runoff on subcatchment imprevious.", rename_labels=True, x_name="Imprevious [%]", y_name="Runoff [m3]"),
        "plot_area": plot(path=os.path.join(settings.BASE_DIR, "data", "df_area.xlsx"), x="area", xaxes=False, title="Dependence of runoff on subcatchment area.", rename_labels=True, x_name="Area [ha]", y_name="Runoff [m3]"),
        "plot_width": plot(path=os.path.join(settings.BASE_DIR, "data", "df_width.xlsx"), x="width", xaxes=True,  stop=1000, title="Dependence of runoff on subcatchment width.", rename_labels=True, x_name="Width [m]", y_name="Runoff [m3]"),
        "manning_impervious": plot(path=os.path.join(settings.BASE_DIR, "data", "df_n_impervious.xlsx"), x="N-Imperv", xaxes=False,  title="Dependence of runoff on Manning's impervious.", rename_labels=True, x_name="Manning's N-Imperv [-]", y_name="Runoff [m3]"),
        "manning_pervious": plot(path=os.path.join(settings.BASE_DIR, "data", "df_n_perv.xlsx"), x="N-Perv", xaxes=False,  title="Dependence of runoff on Manning's pervious.", rename_labels=True, x_name="Manning's N-Perv [-]", y_name="Runoff [m3]"),
        "destore_impervious": plot(path=os.path.join(settings.BASE_DIR, "data", "df_s_imperv.xlsx"), x="Destore-Imperv", xaxes=False,  title="Dependence of runoff on Destore impervious.", rename_labels=True, x_name="Destore Impervious [inches]", y_name="Runoff [m3]"),
        "destore_pervious": plot(path=os.path.join(settings.BASE_DIR, "data", "df_s_perv.xlsx"), x="Destore-Perv", xaxes=False,  title="Dependence of runoff on Destore pervious.", rename_labels=True, x_name="Destore Pervious [inches]", y_name="Runoff [m3]"),
        "zero_impervious": plot(path=os.path.join(settings.BASE_DIR, "data", "df_zero_imperv.xlsx"), x="Zero-Imperv", xaxes=False,  title="Dependence of runoff on zero impervious area.", rename_labels=True, x_name="Zero Impervious [-]", y_name="Runoff [m3]"),
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

@login_required
def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        filename, file_extension = os.path.splitext(uploaded_file.name)

        if file_extension.lower() == '.inp':
            file_path = os.path.join('uploaded_files', filename + file_extension)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Przechowuj ścieżkę do pliku w sesji
            request.session['uploaded_file_path'] = file_path

            return JsonResponse({'message': 'File was sent.'})
        else:
            return JsonResponse({'error': 'Invalid file type. Please upload a .inp file.'})
    return JsonResponse({'error': 'Error occurred while sending file.'})


def get_feature_name(method_name):
    feature_map = {
        'simulate_percent_slope': "PercSlope",
        'simulate_area': "Area",
        'simulate_width': "Width",
        'simulate_percent_impervious': "PercImperv",
        'simulate_percent_zero_imperv': "Zero-Imperv",
    }
    return feature_map.get(method_name, '')

def save_output_file(request, df, output_file_name):
    output_file_path = 'output_files/simulation_result.xlsx'
    df.to_excel(output_file_path, index=False)

    with open(output_file_path, 'rb') as file:
        default_storage.save(output_file_name, file)

    output_file_url = default_storage.url(output_file_name)
    request.session['output_file_url'] = output_file_url
    request.session['output_file_name'] = output_file_name

def get_session_variables(request):
    return {
        'show_download_button': request.session.get('show_download_button', False),
        'user_plot': request.session.get('user_plot', None),
        'results_data': request.session.get('results_data', []),
        'feature_name': request.session.get('feature_name', ''),
        'output_file_name': request.session.get('output_file_name', None),
        'output_file_url': request.session.get('output_file_url', None),
    }


def clear_session_variables(request):
    for variable in ['show_download_button', 'user_plot', 'results_data', 'feature_name', 'output_file_name', 'output_file_url']:  # Dodaj 'output_file_url'
        if variable in request.session:
            del request.session[variable]


@login_required
def simulation_view(request):
    show_download_button = False
    user_plot = None
    output_file_name = None

    if request.method == 'POST':
        form = SimulationForm(request.POST)
        if form.is_valid():
            method_name = form.cleaned_data['option']
            start = form.cleaned_data['start']
            stop = form.cleaned_data['stop']
            step = form.cleaned_data['step']
            catchment_name = form.cleaned_data['catchment_name']

            uploaded_file_path = request.session.get('uploaded_file_path', os.path.abspath('catchment_simulation/example.inp'))

            model = FeaturesSimulation(subcatchment_id=catchment_name, raw_file=uploaded_file_path)
            feature_name = get_feature_name(method_name)

            method = getattr(model, method_name)
            df = method(start=start, stop=stop, step=step)
            df = df[[feature_name, 'runoff']]

            show_download_button = True

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_name = f"{request.user.username}_simulation_result_{timestamp}.xlsx"

            user_plot = plot(df=df, x=feature_name, xaxes=False, title=f"Dependence of runoff on subcatchment {feature_name}.")

            request.session['show_download_button'] = show_download_button
            request.session['user_plot'] = user_plot
            request.session['results_data'] = [(row[feature_name], row['runoff']) for index, row in df.iterrows()]
            request.session['feature_name'] = feature_name

            save_output_file(request, df, output_file_name)

            return redirect('main:simulation')
        
    else:
        form = SimulationForm()
        session_data = get_session_variables(request)
        clear_session_variables(request)

    return render(request, 'main/simulation.html', {'form': form, **session_data, 'output_file_name': output_file_name})

def download_result(request):
    output_file_path = 'output_files/simulation_result.xlsx'

    if os.path.exists(output_file_path):
        response = FileResponse(open(output_file_path, 'rb'), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=simulation_result.xlsx'
        return response
    else:
        return HttpResponseNotFound("File not found.")
