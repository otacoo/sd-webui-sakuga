(function () {
    /** temporary fix for https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/668
     *  @see https://github.com/gradio-app/gradio/issues/1721 */
    let resizeFrame = null;

    function imageMaskResize() {
        const tab = gradioApp().getElementById('tab_img2img');
        if (!tab || tab.style.display === 'none') return;

        const canvases = gradioApp().querySelectorAll('#img2maskimg .touch-none canvas');
        if (!canvases.length) return;

        const wrapper = canvases[0].closest('.touch-none');
        const previewImage = wrapper.previousElementSibling;

        if (!previewImage.complete) {
            if (!previewImage._maskFixLoading) {
                previewImage._maskFixLoading = true;
                previewImage.addEventListener('load', function () {
                    previewImage._maskFixLoading = false;
                    imageMaskResize();
                }, {once: true});
            }
            return;
        }

        const w = previewImage.width;
        const h = previewImage.height;
        const nw = previewImage.naturalWidth;
        const nh = previewImage.naturalHeight;
        const key = `${w}x${h}:${nw}x${nh}`;

        if (wrapper._maskFixKey === key) return;
        wrapper._maskFixKey = key;

        const portrait = nh > nw;

        const wW = Math.min(w, portrait ? h / nh * nw : w / nw * nw);
        const wH = Math.min(h, portrait ? h / nh * nh : w / nw * nh);

        wrapper.style.width = `${wW}px`;
        wrapper.style.height = `${wH}px`;
        wrapper.style.left = `0px`;
        wrapper.style.top = `0px`;

        canvases.forEach(c => {
            c.style.width = c.style.height = '';
            c.style.maxWidth = '100%';
            c.style.maxHeight = '100%';
            c.style.objectFit = 'contain';
        });
    }

    function scheduleImageMaskResize() {
        if (resizeFrame) return;
        resizeFrame = requestAnimationFrame(function () {
            resizeFrame = null;
            imageMaskResize();
        });
    }

    onAfterUiUpdate(scheduleImageMaskResize);
    window.addEventListener('resize', scheduleImageMaskResize);
})();
