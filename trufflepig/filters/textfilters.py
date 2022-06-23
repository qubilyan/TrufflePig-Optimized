import re
import logging


logger = logging.getLogger(__name__)


def filter_html_tags(text):
    return re.sub('</?[a-zA-Z]{1,11}>', '', text)


def filter_images_and_links(text):
    # filter images altogether
    text =  re.sub('!\[[-a-zA-Z0-9?@: %._\+~#=/()]*\]\([-a-zA-Z0-9?@:%._\+~#=/()]+\)', '', text)
    # replace the links just with the name
    text =  re.sub('\[([-a-zA-Z0-9?@: %._\+~#=/()]*)\]\([-a-zA-Z0-9?@:%._\+~#=/()]+\)', '\g<1>', text)
    return text


def get_image_urls(text):
    images = re.findall('!\[[-a-zA-Z0-9?@: %._\+~#=/()]*\]\([-a-zA-Z0-9?@:%._\+~#=/()]+\)|<img[^>]+src="[^">]+"[^>]*>', text)
    image_urls = []
    for image in images:
        if image.startswith('<img'):
            image_url = re.sub('<img[^>]+src="([^">]+)"[^>]*>',  '\g<1>', image)
        else:
            image_url = re.sub('!\[[-a-zA-Z0-9?@: %._\+~#=/()]*\]\(([-a-zA-Z0-9?@:%._\+~#=/()]+)\)', '\g<1>', ima