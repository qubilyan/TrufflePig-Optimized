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
            image_url = re.sub('!\[[-a-zA-Z0-9?@: %._\+~#=/()]*\]\(([-a-zA-Z0-9?@:%._\+~#=/()]+)\)', '\g<1>', image)
        image_urls.append(image_url)
    return image_urls


def filter_urls(text):
    return re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]'
                   '[a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.'
                   '[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}'
                   '|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]'
                   '{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})', '', text)


def filter_special_characters(text):
    return re.sub('[^A-Za-z0-9\s;,.?!]+', '', text)


EXPRESSIONS = (
    '&?nbsp',
    'aligncenter',
    'styletextalign',
    'href',
    'img',
    'src',
    'div',
    """class=["']text-justify["']""",
    'h[1-6]',
    'hspace[0-9]*',
    'alignleft',
    'alignright',
    'border[0-9]+',
    'height[0-9]+',
    'width[0-9]+',
)


def filter_formatting(text):
    for expression in EXPRESSIONS:
        text = re.sub(expression, '', text)
        text = re.sub(expression.upper(), '', text)
        text = re.sub(expression.capitalize(), '', text)

    return text


def fil