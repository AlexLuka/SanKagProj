FROM python:3.6

# Add Tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini

COPY santander/ /opt/santander/
COPY setup.py /opt


WORKDIR /opt


RUN python setup.py sdist           \
 && pip install dist/santander*     \
 && rm -r setup.py dist santander   \
 && chmod +x /tini


ENTRYPOINT ["/tini", "--"]

CMD ["python", "-m", "santander.start"]
