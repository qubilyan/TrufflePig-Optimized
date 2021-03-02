import trufflepig.filters.textfilters as tptf


def test_filter_html_tags():
    result = tptf.filter_html_tags('<jkdjdksakd>hi</img>')
    assert result == 'hi'


def test_filter_images_and_links():
    result = tptf.filter_images_and_links('Lookat ![j kjds](wehwjrkjewrk.de), yes [iii](jlkajddjsla), and '
                        '![images (17).jpg](https://steemitimages.com/DQmQF5BxHtPdPu1yKipV67GpnRdzemPpEFCqB59kVXC6Ahy/images%20(1