/**
 * Color Management Module
 * Handles color palette generation, custom color schemes, and color conversions
 */

(function() {
    'use strict';

    window.pciSeq = window.pciSeq || {};
    const state = window.pciSeq.state;

    // Zero is the "nothing here" class. It always gets the same colour, whatever the
    // taxonomy is and whatever colour scheme gets loaded, so it reads the same on every map.
    const ZERO_CLASS = 'Zero';
    const ZERO_COLOR = [0, 0, 0];

    // a class the loaded colour scheme says nothing about. Same grey as the fallback in
    // rendering.js, so "no colour" looks the same wherever it comes from.
    const UNSET_COLOR = [128, 128, 128];

    // how many colours to make before the class names are known
    const DEFAULT_NUM_COLORS = 65;

    function paletteColor(k, hueStep) {
        const hue = (k * hueStep) % 360;
        const saturation = 70 + (k % 3) * 10; // Vary saturation slightly
        const lightness = 50 + (k % 2) * 10;  // Vary lightness slightly
        return hslToRgb(hue, saturation, lightness);
    }

    function paintZero() {
        Object.entries(state.cellClassNames).forEach(([idx, name]) => {
            if (name === ZERO_CLASS) state.cellClassColors[parseInt(idx)] = ZERO_COLOR;
        });
    }

    // One colour per real class, spread evenly round the hue wheel.
    //
    // It used to make 65 colours and stop, so with a bigger taxonomy every class from
    // the 66th on came out in the grey fallback. Now it is called again once the class
    // names have arrived and makes as many colours as there are real classes. Zero is
    // left out of the count and painted black. Anything that still ends up without a
    // colour gets the grey fallback in rendering.js, same as before.
    function generateColorPalette(classNames) {
        // wipe it in place, other modules hold on to this object
        Object.keys(state.cellClassColors).forEach((k) => delete state.cellClassColors[k]);

        if (!classNames || classNames.length === 0) {
            const hueStep = 360 / DEFAULT_NUM_COLORS;
            for (let i = 0; i < DEFAULT_NUM_COLORS; i++) {
                state.cellClassColors[i] = paletteColor(i, hueStep);
            }
            return;
        }

        const realClasses = [];
        classNames.forEach((name, idx) => {
            if (name !== ZERO_CLASS) realClasses.push(idx);
        });
        const hueStep = 360 / Math.max(realClasses.length, 1);
        realClasses.forEach((classIdx, k) => {
            state.cellClassColors[classIdx] = paletteColor(k, hueStep);
        });
        paintZero();
    }

    // HSL to RGB conversion (use d3 for clarity and reliability)
    function hslToRgb(h, s, l) {
        const rgb = d3.hsl(h, s / 100, l / 100).rgb();
        return [
            Math.round(rgb.r),
            Math.round(rgb.g),
            Math.round(rgb.b)
        ];
    }

    // Convert hex color to RGB using D3.js
    function hexToRgb(hex) {
        const color = d3.rgb(hex);
        return [color.r, color.g, color.b];
    }

    // Apply custom color scheme to cell classes
    function applyColorScheme(colorScheme) {
        // remembered so it can go back on when the palette is rebuilt for a new run
        state.customColorScheme = colorScheme;
        let appliedCount = 0;
        const notFoundClasses = [];

        // Create reverse mapping: class name -> class index
        const nameToIndex = {};
        Object.entries(state.cellClassNames).forEach(([idx, name]) => {
            nameToIndex[name] = parseInt(idx);
        });

        // Once a scheme is loaded the colours are the user's. A class the scheme does not
        // name is shown grey, so it is obvious it was left out. It used to keep whatever
        // the auto palette had given it, which could pass for one of the chosen colours.
        Object.values(nameToIndex).forEach((classIdx) => {
            state.cellClassColors[classIdx] = UNSET_COLOR;
        });

        // Apply custom colors
        Object.entries(colorScheme).forEach(([className, hexColor]) => {
            const classIdx = nameToIndex[className];

            if (classIdx !== undefined) {
                try {
                    const rgb = hexToRgb(hexColor);
                    state.cellClassColors[classIdx] = rgb;
                    appliedCount++;
                } catch (err) {
                    console.warn(`Invalid color format for ${className}: ${hexColor}`);
                }
            } else {
                notFoundClasses.push(className);
            }
        });

        if (notFoundClasses.length > 0) {
            console.warn(`Classes not found in data: ${notFoundClasses.join(', ')}`);
        }

        // the other way round: classes in the data that the scheme left out, shown grey
        const named = new Set(Object.keys(colorScheme));
        const unnamed = Object.values(state.cellClassNames)
            .filter((name) => name !== ZERO_CLASS && !named.has(name));
        if (unnamed.length > 0) {
            console.warn(`Classes with no colour in the scheme, shown grey: ${unnamed.join(', ')}`);
        }

        // Zero is black whatever the scheme says
        paintZero();

        return { appliedCount, notFoundClasses };
    }

    // Load custom colors from user-provided scheme
    function loadCustomColors(colorScheme) {
        const statusEl = document.getElementById('file-status');

        // Check if class names have been loaded yet
        const hasClassNames = Object.keys(state.cellClassNames).length > 0;

        if (!hasClassNames) {
            // Store color scheme for later application
            state.pendingColorScheme = colorScheme;
            statusEl.textContent = `Color scheme loaded (${Object.keys(colorScheme).length} classes). Will apply when algorithm starts.`;
            statusEl.className = 'file-status success';

            // Clear status after 5 seconds
            setTimeout(() => {
                statusEl.textContent = '';
                statusEl.className = 'file-status';
            }, 5000);
            return;
        }

        // Apply colors immediately
        const { appliedCount, notFoundClasses } = applyColorScheme(colorScheme);

        // Update UI
        if (appliedCount > 0) {
            statusEl.textContent = `Applied ${appliedCount} custom colors`;
            statusEl.className = 'file-status success';

            // Refresh legend and visualization
            window.pciSeq.updateLegend();
            window.pciSeq.render();
        } else {
            statusEl.textContent = 'No matching classes found';
            statusEl.className = 'file-status error';
        }

        // Clear status after 5 seconds
        setTimeout(() => {
            statusEl.textContent = '';
            statusEl.className = 'file-status';
        }, 5000);
    }

    // Handle color file upload
    function handleColorFileUpload(event) {
        const file = event.target.files[0];
        const statusEl = document.getElementById('file-status');

        if (!file) return;

        statusEl.textContent = 'Loading...';
        statusEl.className = 'file-status';

        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const colorScheme = JSON.parse(e.target.result);

                // Validate JSON structure
                if (typeof colorScheme !== 'object' || Array.isArray(colorScheme)) {
                    throw new Error('Invalid JSON format. Expected object with class_name: hex_color pairs');
                }

                loadCustomColors(colorScheme);
            } catch (err) {
                statusEl.textContent = `Error: ${err.message}`;
                statusEl.className = 'file-status error';
                console.error('Failed to load color scheme:', err);
            }
        };

        reader.onerror = () => {
            statusEl.textContent = 'Failed to read file';
            statusEl.className = 'file-status error';
        };

        reader.readAsText(file);

        // Reset file input so same file can be uploaded again
        event.target.value = '';
    }

    // Export functions
    window.pciSeq.colors = {
        generateColorPalette: generateColorPalette,
        applyColorScheme: applyColorScheme,
        loadCustomColors: loadCustomColors,
        handleColorFileUpload: handleColorFileUpload
    };

})();
