import re
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import time
import json

from functools import reduce
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import date, datetime

from urllib.error import HTTPError
from urllib.request import Request, urlopen

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"

BASE_URL = "https://www.icarros.com.br/ache/listaanuncios.jsp?bid=1&pag={page}&lis=0&ord=24&sop=sta_1.1_-cid_3632.1_-esc_2.1_-rai_0.1_"

N_BY_PAGE = 20

SLEEP_TIME_MEAN = 7

SLEEP_TIME_STD = 5

SLEEP_TIME_BIAS = 7

OUTPUT_PATH = "data/raw/scrapped/"


def run_scrapper():

    qt = get_amount_cars()

    logging.info(f"Total de carros: {qt}")

    max_page = int(qt / N_BY_PAGE) + 1

    logging.info(f"Última página: {max_page}")

    results = pd.DataFrame()

    for page in tqdm(range(1, 1 + max_page)):

        try:

            today = datetime.now().date()

            time.sleep(simulated_time(SLEEP_TIME_MEAN, SLEEP_TIME_STD, SLEEP_TIME_BIAS))

            url = BASE_URL.format(page=page)

            response = get_page(url)

            soup = BeautifulSoup(response, "html.parser")

            script_listings = soup.find("script", {"type": "application/ld+json"})

            script_listings_json = json.loads(script_listings.text)

            scrip_listing = pd.DataFrame(
                map(get_script_listing_element, script_listings_json["itemListElement"])
            )

            html_listings = soup.findAll(name="li", class_="anuncio")

            html_listing = pd.DataFrame(map(get_html_listing_element, html_listings))

            df = html_listing.merge(scrip_listing, on="id").assign(
                reference_date=today, page=page
            )

            results = pd.concat([results, df])

            table = pa.Table.from_pandas(df)

            pq.write_to_dataset(
                table,
                root_path=OUTPUT_PATH,
                partition_cols=["reference_date", "page"],
                existing_data_behavior="delete_matching",
            )

        except Exception as err:
            logging.error(f"Error on page {page}" + err.__str__())


def get_page(url: str, timeout: int = 20, verbose: int = 0):
    """Make a request to a site html and returns the html code

    :param url: URL from the desired site
    :type url: str
    :param timeout: Maximum time in seconds to wait the response, defaults to 20
    :type timeout: int, optional
    :param verbose: Logging level, defaults to 0
    :type verbose: int, optional
    :return: The site htto response
    :rtype: http.client.HTTPResponse
    """

    request = Request(url)

    request.add_header("User-Agent", USER_AGENT)

    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as e:
        if verbose > 0:
            print("[error]", e)

        if e.getcode() == 400:
            response = None
        elif e.getcode() == 404:
            response = None

    return response


def get_amount_cars():

    page = 1
    url = BASE_URL.format(page=page)
    response = get_page(url)
    soup = BeautifulSoup(response, "html.parser")
    script_listings = soup.find("script", {"type": "application/ld+json"})
    script_listings_json = json.loads(script_listings.text)
    qt = int(script_listings_json["description"].split()[0])

    assert isinstance(qt, int)

    return qt


def get_script_listing_element(x):

    url_str = x["item"]["url"]

    url_elements = url_str.split("/")

    brand_lower = url_elements[5]

    car_name = url_elements[6]

    try:
        id_ = int(url_elements[-1][1:])
    except:
        id_ = None

    return {
        "position": x["position"],
        "id": id_,
        "car": car_name,
        "brand": brand_lower,
        "brand_full": x["item"]["brand"]["name"],
        "car_description": x["item"]["name"],
        "price": x["item"]["offers"]["price"],
        "seller": x["item"]["offers"]["seller"]["name"],
        "image": x["item"]["image"],
        "url": x["item"]["url"],
    }


def get_html_listing_element(li):

    results = {}

    try:
        results.update({"id": int(li.get("id")[2:])})
    except:
        results.update({"id": -1})

    try:
        results.update({"preco": li.find("h3", class_="preco_anuncio").text})
    except:
        pass

    try:
        primeiro_key = li.find("li", class_="primeiro").find("span").text
        primeiro_value = li.find("li", class_="primeiro").find("p").text
        primeiro_value = re.findall(r"[\w']+", primeiro_value)[0]
        results.update({primeiro_key: primeiro_value})
    except:
        pass

    try:
        zerokm = li.find("li", class_="zerokm").text
        zerokm = re.findall(r"[\w']+", zerokm)[0][:-2]
        results.update({"km": zerokm})
    except:
        try:
            usado_key = li.find("li", class_="usado").find("span").text
            usado_value = li.find("li", class_="usado").find("p").text
            usado_value = "".join(re.findall(r"[\w']+", usado_value))
            results.update({"km": usado_value})
        except:
            pass

    try:
        description = li.find("p", class_="texto_padrao").text
        results.update({"description": description})
    except:
        pass

    return results


def simulated_time(mu: float, std: float, val: float):
    sleep_time = np.random.normal(mu, std, 1)
    sleep_time = np.where(sleep_time > 0, sleep_time, 0)

    t = 0
    t = val * np.random.random()
    t = np.where(t > val - 1, val, 0)

    sleep_time = sleep_time + t

    return sleep_time[0]


if __name__ == "__main__":
    run_scrapper()