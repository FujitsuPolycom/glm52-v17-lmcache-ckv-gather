ARG V17_BASE_IMAGE=voipmonitor/vllm:fathomless-firmament-v17-vllm05f50ae-b12x1377d5f-fi801d57a-cu132-20260715@sha256:9b6f1ab6db4d3a7b7b786481eb32abe82e86d185648d62c3ac1cfa6d72a55e47
FROM ${V17_BASE_IMAGE}

ARG RELEASE_VERSION=0.1.0-exp.2
ARG B12X_BASE_SHA=928741029c07dfb05aca12310499e8562963dc24a15a50597832ce04916eecfb
ARG MLA_BASE_SHA=2868c27eaea476bf49d991a18e752d1920c7ee334f53d11b417e1255746b815b
ARG INDEXER_BASE_SHA=27848bf7c77b0c7b50edda3c4794cb6c85ce34d897165c5df741d88883b11838
ARG VLLM_SITE=/opt/venv/lib/python3.12/site-packages/vllm
ARG VLLM_SRC=/opt/vllm/vllm

LABEL org.opencontainers.image.title="GLM-5.2 v17 LMCache + CKV gather" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.description="Experimental TP4/DCP4 GLM-5.2 v17 image with RAM LMCache and eager-prefill CKV gather" \
      io.glm52.release.status="gpu-validated-experimental"

USER root

RUN /opt/venv/bin/pip install --no-cache-dir "lmcache==0.4.6"

COPY base/patches/ /opt/glm52-release/lmcache-patches/
COPY base/validate_install.py /opt/glm52-release/validate_lmcache.py
COPY base/start-v17-lmcache.sh /opt/glm52-release/start-v17-lmcache.sh
COPY base/verify_reuse.py /opt/glm52-release/verify_reuse.py
COPY base/benchmark_reuse.py /opt/glm52-release/benchmark_reuse.py
COPY base/smoke-cache-daemon.sh /opt/glm52-release/smoke-cache-daemon.sh

RUN chmod 0755 /opt/glm52-release/start-v17-lmcache.sh \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_kv_xfer_assert_v10.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_dcp_lmcache.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_dcp_mla_store.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_ipc_event_lifetime.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_chunked_store_futures.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_preempt_safe_restore.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_v17_launcher.py \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_instanttensor_mtp_fallback.py \
    && before="$(sha256sum \
        /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py \
        /opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py \
        /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/lookup.py | sha256sum)" \
    && /opt/venv/bin/python /opt/glm52-release/lmcache-patches/patch_dcp_lmcache.py \
    && after="$(sha256sum \
        /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py \
        /opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py \
        /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/lookup.py | sha256sum)" \
    && test "${before}" = "${after}" \
    && /opt/venv/bin/python /opt/glm52-release/validate_lmcache.py \
    && /opt/venv/bin/pip check

RUN test "$(sha256sum ${VLLM_SITE}/v1/attention/backends/mla/b12x_mla_sparse.py | cut -d' ' -f1)" = "${B12X_BASE_SHA}" \
    && test "$(sha256sum ${VLLM_SITE}/model_executor/layers/attention/mla_attention.py | cut -d' ' -f1)" = "${MLA_BASE_SHA}" \
    && test "$(sha256sum ${VLLM_SITE}/model_executor/layers/sparse_attn_indexer.py | cut -d' ' -f1)" = "${INDEXER_BASE_SHA}"

COPY gather/overlay/ /opt/glm52-release/ckv-overlay/
COPY gather/validate_install.py /opt/glm52-release/validate_ckv_gather.py
COPY gather/test_ckv_gather_runtime.py /opt/glm52-release/test_ckv_gather_runtime.py
COPY gather/patch_eager_launcher.py /opt/glm52-release/patch_eager_launcher.py

RUN install -m 0644 \
        /opt/glm52-release/ckv-overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
        ${VLLM_SITE}/v1/attention/backends/mla/b12x_mla_sparse.py \
    && install -m 0644 \
        /opt/glm52-release/ckv-overlay/vllm/model_executor/layers/attention/mla_attention.py \
        ${VLLM_SITE}/model_executor/layers/attention/mla_attention.py \
    && install -m 0644 \
        /opt/glm52-release/ckv-overlay/vllm/model_executor/layers/sparse_attn_indexer.py \
        ${VLLM_SITE}/model_executor/layers/sparse_attn_indexer.py \
    && install -m 0644 \
        /opt/glm52-release/ckv-overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
        ${VLLM_SRC}/v1/attention/backends/mla/b12x_mla_sparse.py \
    && install -m 0644 \
        /opt/glm52-release/ckv-overlay/vllm/model_executor/layers/attention/mla_attention.py \
        ${VLLM_SRC}/model_executor/layers/attention/mla_attention.py \
    && install -m 0644 \
        /opt/glm52-release/ckv-overlay/vllm/model_executor/layers/sparse_attn_indexer.py \
        ${VLLM_SRC}/model_executor/layers/sparse_attn_indexer.py \
    && /opt/venv/bin/python /opt/glm52-release/patch_eager_launcher.py \
    && /opt/venv/bin/python /opt/glm52-release/validate_ckv_gather.py \
    && /opt/venv/bin/python -m compileall -q \
        ${VLLM_SITE}/v1/attention/backends/mla/b12x_mla_sparse.py \
        ${VLLM_SITE}/model_executor/layers/attention/mla_attention.py \
        ${VLLM_SITE}/model_executor/layers/sparse_attn_indexer.py

ENTRYPOINT ["/opt/glm52-release/start-v17-lmcache.sh"]
