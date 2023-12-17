from django.shortcuts import render, HttpResponse
from .ocr import test
# Create your views here.
def page_1(request):
    return render(request, 'page_1.html')

def page_2(request):
    if request.method == "POST":
        url = request.POST.get('url')
        if(url!=""):
            test.process_video_stream(url)
            print(url)
        else:
            return ("There is no url")


    
    count=test.v_count()
    return render(request, 'page_2.html',{'count':count})
