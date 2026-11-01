from django.urls import path
from views import home, download_notes, result_image
urlpatterns = [path('', home, name='home'), path('notes/<str:job>/', download_notes, name='notes'), path('image/<str:job>/<str:kind>/', result_image, name='image')]
