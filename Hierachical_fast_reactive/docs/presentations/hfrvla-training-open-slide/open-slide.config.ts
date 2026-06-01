import type { OpenSlideConfig } from '@open-slide/core';
import { zhTW } from '@open-slide/core/locale';

const openSlideConfig: OpenSlideConfig = {
  locale: zhTW,
  build: {
    showSlideBrowser: false,
    showSlideUi: true,
    allowHtmlDownload: false,
  },
};

export default openSlideConfig;
