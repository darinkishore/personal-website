import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://darinkishore.com", // If you don't have a domain yet, use "https://darinkishore.github.io" or similar
  author: "Darin Kishore",
  authorShort: "Darin",
  desc: "Personal website and blog of Darin Kishore - Thoughts on technology, programming, and personal projects",
  title: "Darin Kishore",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  profile: "/profile.jpg",
  postPerIndex: 4,
  postPerPage: 5,
  scheduledPostMargin: 15 * 60 * 1000,
  showArchives: true,
};

export const LOCALE = {
  lang: "en", // html lang code. Set this empty and default will be "en"
  langTag: ["en-EN"], // BCP 47 Language Tags. Set this empty [] to use the environment default
} as const;

export const LOGO_IMAGE = {
  enable: true,
  svg: true, // set to false if using PNG
  width: 216,
  height: 46,
};

// Keep only the social media you actually use
export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/darinkishore",
    linkTitle: `${SITE.title} on Github`,
    active: true,
  },
  {
    name: "LinkedIn",
    href: "https://linkedin.com/in/darin-kishore-a65576a1",
    linkTitle: `${SITE.title} on LinkedIn`,
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:darinkishore@protonmail.com",
    linkTitle: `Send an email to ${SITE.title}`,
    active: true,
  }
];
