import re
import logging


logger = logging.getLogger(__name__)


def filter_html_tags(text):
    return re.sub('</?[a-zA-Z]{1,11}>', '', text)


def filter_images_and_links(text):
    # filter images altogether
    text =  re.sub('!\[[-a-zA-Z