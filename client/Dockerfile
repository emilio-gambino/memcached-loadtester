FROM cloudsuite/base-os:ubuntu

# add our user and group first to make sure their IDs get assigned consistently, regardless of whatever dependencies get added
RUN groupadd -r memcache && useradd -r -g memcache memcache

ENV DEBIAN_FRONTEND noninteractive

RUN buildDeps='curl gcc g++ libc6-dev make' \
	&& set -x \
	&& apt-get update && apt-get install -y $buildDeps libevent-dev vim --no-install-recommends \
	&& rm -rf /var/lib/apt/lists/* \
	&& curl -k -L --remote-name https://github.com/emilio-gambino/memcached-loadtester/archive/refs/tags/v2.tar.gz \
	&& mkdir -p /usr/src/memcached \
	&& tar xvf v2.tar.gz -C /usr/src/memcached --strip-components=1 \
	&& rm v2.tar.gz \
	&& cd /usr/src/memcached/memcached_client \
	&& make -j $(nproc) \
	&& chown -R memcache:memcache /usr/src/memcached \
	&& apt-get purge -y --auto-remove $buildDeps

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

#USER memcache
CMD ["/bin/bash"]
