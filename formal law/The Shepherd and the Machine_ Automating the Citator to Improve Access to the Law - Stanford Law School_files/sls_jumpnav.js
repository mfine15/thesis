jQuery(document).ready(function ($) {

    var panels = modern_tribe.panels.core;

    if (typeof panels.$jump_nav === 'undefined' || typeof panels.$chapter_nav === 'undefined') {
        return;
    }

    // show preview of jump nav and then close it
    setTimeout(function() {
        panels._show_chapter_nav();

        setTimeout(function() {
            panels._hide_chapter_nav();
        }, 2000);
    }, panels.options.jump_nav_delay);

});
