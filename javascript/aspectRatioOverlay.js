(function () {
    let currentWidth = null;
    let currentHeight = null;
    let arFrameTimeout = null;
    let arFrame = null;
    let widthInput = null;
    let heightInput = null;

    function dimensionChange(e, is_width, is_height) {
        if (is_width) {
            currentWidth = e.target.value * 1.0;
        }
        if (is_height) {
            currentHeight = e.target.value * 1.0;
        }

        if (arFrame) return;
        arFrame = requestAnimationFrame(function () {
            arFrame = null;
            updateOverlay();
        });
    }

    function updateOverlay() {
        let tabImg2img = gradioApp().querySelector("#tab_img2img");
        if (!tabImg2img || tabImg2img.style.display != "block") {
            return;
        }

        let targetElement = null;

        let tabIndex = get_tab_index('mode_img2img');
        if (tabIndex == 0) { // img2img
            targetElement = gradioApp().querySelector('#img2img_image div[data-testid=image] img');
        } else if (tabIndex == 1) { //Sketch
            targetElement = gradioApp().querySelector('#img2img_sketch div[data-testid=image] img');
        } else if (tabIndex == 2) { // Inpaint
            targetElement = gradioApp().querySelector('#img2maskimg div[data-testid=image] img');
        } else if (tabIndex == 3) { // Inpaint sketch
            targetElement = gradioApp().querySelector('#inpaint_sketch div[data-testid=image] img');
        }

        if (!targetElement) return;

        let arPreviewRect = gradioApp().querySelector('#imageARPreview');
        if (!arPreviewRect) {
            arPreviewRect = document.createElement('div');
            arPreviewRect.id = "imageARPreview";
            gradioApp().appendChild(arPreviewRect);
        }

        let viewportOffset = targetElement.getBoundingClientRect();

        let viewportscale = Math.min(targetElement.clientWidth / targetElement.naturalWidth, targetElement.clientHeight / targetElement.naturalHeight);

        let scaledx = targetElement.naturalWidth * viewportscale;
        let scaledy = targetElement.naturalHeight * viewportscale;

        let clientRectTop = (viewportOffset.top + window.scrollY);
        let clientRectLeft = (viewportOffset.left + window.scrollX);
        let clientRectCentreY = clientRectTop + (targetElement.clientHeight / 2);
        let clientRectCentreX = clientRectLeft + (targetElement.clientWidth / 2);

        let arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
        let arscaledx = currentWidth * arscale;
        let arscaledy = currentHeight * arscale;

        let arRectTop = clientRectCentreY - (arscaledy / 2);
        let arRectLeft = clientRectCentreX - (arscaledx / 2);
        let arRectWidth = arscaledx;
        let arRectHeight = arscaledy;

        arPreviewRect.style.top = arRectTop + 'px';
        arPreviewRect.style.left = arRectLeft + 'px';
        arPreviewRect.style.width = arRectWidth + 'px';
        arPreviewRect.style.height = arRectHeight + 'px';

        clearTimeout(arFrameTimeout);
        arFrameTimeout = setTimeout(function () {
            arPreviewRect.style.display = 'none';
        }, 2000);

        arPreviewRect.style.display = 'block';
    }

    onAfterUiUpdate(function () {
        let arPreviewRect = gradioApp().querySelector('#imageARPreview');
        if (arPreviewRect) {
            arPreviewRect.style.display = 'none';
        }

        let tabImg2img = gradioApp().querySelector("#tab_img2img");
        if (!tabImg2img || tabImg2img.style.display != "block") {
            return;
        }

        let width = gradioApp().querySelector('#img2img_width input');
        let height = gradioApp().querySelector('#img2img_height input');

        if (width && width !== widthInput) {
            width.addEventListener('input', function (e) {
                dimensionChange(e, true, false);
            });
            widthInput = width;
        }
        if (height && height !== heightInput) {
            height.addEventListener('input', function (e) {
                dimensionChange(e, false, true);
            });
            heightInput = height;
        }

        if (width) {
            currentWidth = width.value * 1.0;
        }
        if (height) {
            currentHeight = height.value * 1.0;
        }
    });
})();
