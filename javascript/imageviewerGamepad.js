(function () {
    let gamepads = [];

    function lightboxOpen() {
        const modal = gradioApp().getElementById('lightboxModal');
        return modal && modal.style.display === 'flex';
    }

    function pollGamepad(index, event, state) {
        if (state.stopped) return;

        if (typeof opts === 'undefined' || !opts.js_modal_lightbox_gamepad || !lightboxOpen()) {
            const delay = (typeof opts !== 'undefined' && opts.js_modal_lightbox_gamepad) ? 250 : 2000;
            state.timer = setTimeout(function () {
                pollGamepad(index, event, state);
            }, delay);
            return;
        }

        const gamepad = navigator.getGamepads()[index];
        if (!gamepad) {
            state.timer = setTimeout(function () {
                pollGamepad(index, event, state);
            }, 500);
            return;
        }

        const now = performance.now();
        const xValue = gamepad.axes[0];
        const repeat = opts.js_modal_lightbox_gamepad_repeat || 250;

        if (state.waiting) {
            if ((xValue > -0.3 && xValue < 0.3) || now >= state.waitUntil) {
                state.waiting = false;
            }
        }

        if (!state.waiting) {
            if (xValue <= -0.3) {
                modalPrevImage(event);
                state.waiting = true;
                state.waitUntil = now + repeat;
            } else if (xValue >= 0.3) {
                modalNextImage(event);
                state.waiting = true;
                state.waitUntil = now + repeat;
            }
        }

        state.frame = requestAnimationFrame(function () {
            pollGamepad(index, event, state);
        });
    }

    window.addEventListener('gamepadconnected', (e) => {
        const index = e.gamepad.index;
        const state = {
            stopped: false,
            timer: null,
            frame: null,
            waiting: false,
            waitUntil: 0
        };
        gamepads[index] = state;
        pollGamepad(index, e, state);
    });

    window.addEventListener('gamepaddisconnected', (e) => {
        const state = gamepads[e.gamepad.index];
        if (!state) return;
        state.stopped = true;
        clearTimeout(state.timer);
        cancelAnimationFrame(state.frame);
        gamepads[e.gamepad.index] = null;
    });

    // Primarily for vr controller type pointer devices.
    // I use the wheel event because there's currently no way to do it properly with web xr.

    let isScrolling = false;
    window.addEventListener('wheel', (e) => {
        if (typeof opts === 'undefined' || !opts.js_modal_lightbox_gamepad || isScrolling || !lightboxOpen()) return;
        isScrolling = true;

        if (e.deltaX <= -0.6) {
            modalPrevImage(e);
        } else if (e.deltaX >= 0.6) {
            modalNextImage(e);
        }

        setTimeout(() => {
            isScrolling = false;
        }, opts.js_modal_lightbox_gamepad_repeat || 250);
    });
})();
