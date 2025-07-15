---
author: Darin Kishore
pubDatetime: 2025-07-08T00:00:00Z
title: The Next Moat
slug: the-next-moat
featured: false
draft: false
tags:
  - ai
  - engineering
  - product
description: When AI capabilities become commoditized, taste and evaluation become the new competitive advantage.
---

## Midjourney

Why hasn't anyone—**literally anyone**—made a better Midjourney? Think about it. ChatGPT image generation, gemini's native image generation, replicate and their full suite of image generation models, BlackForestLabs, Stable Diffusion, **everyone** has tried.

But you'll never get a committed Midjourney stan to switch. Why?

For starters, the UX. It's unique—you type an idea, can be a couple of words, can be a paragraph with lots of specificity, and you're going to get four different images—each one a variation of that idea, each one beautiful and aesthetically pleasing in its own way. You can then choose an image, whichever one speaks to you, and do whatever you want to it! You can "remix" it, creating four different images from you to choose from, and keep going until you find the version of the idea that you love. When you're done and you've created the idea you really wanted, you can then "upscale" it—a final selection, increasing the picture's size, that says "I love this image so much I made it bigger".

**This is why nobody's made a better Midjourney**.

Think about it. What does Midjourney get from having users make all these choices? (_hint: what is MJ known for?_)

They get **taste**.

By having users make choices over and over again, they are able to effectively distill **what their users want**. Midjourney appeals to artists, creatives of all kinds, because their models have incredible aesthetic sensibilities—by using the preferences of their users, they can tune their models in the long run to get even closer to the exact thing that their users want.

They also have a very clear idea of what that is—when a user upscales an image, it's the culmination of all the choices that they made. They can leverage this sequence of decisions and in the next iteration of the model, try and get the users to the image that they would want to upscale faster! It's a really nice, straightforward metric that correlates very precisely with user satisfaction. And the user experience they create brings them to this perfect distillation of taste!

Before Midjourney releases a model, they have users judge between pairs of images for weeks leading up to the release. They take the raw image generation capabilities and refine them into something that is **exactly** what its users want for the new model. They have profiles, which let you pick between pairs of images to distill your own personal style and get even closer to getting the kinds of things you think are amazing on your first image generation.

Midjourney's defensibility, even against more technically skilled (gpt-4o image generation was much more precise at getting exactly what people asked of it) competition, teaches us a key lesson of competing in the AI era—The AI-native companies that win, in the short *and* long run, are the companies that can operationalize taste, providing user experiences that are much more magical than any competition could ever be.

## What Remains Defensible?

AI is getting more and more capable. Most models, given enough compute, are "good enough" to do most AI tasks in most AI products. And almost everyone has access to the smartest models.

So how does anyone differentiate themselves?

Anyone can call an LLM API. Anyone can chain three prompts together. What they _cannot_ copy overnight is **knowing what good looks like in your domain—and operationalizing it**.

This is trickier than it sounds because it's not enough to just know what's good—you have to truly, deeply understand what your customers want (including what they're not telling you, and would never tell you!), in every sense of the word.

By investing resources in _what is hard to copy_: **defining and verifying taste**.

The bits that are hard to copy:

- **Expert annotations**: data labeled by and conversations with experts (eg; for diagrams: senior architects) where you gain much more clarity about what exactly it is that you need to build.
- **Evaluator Craftsmanship**: the combination of rigor, expertise, and great engineering that lets you evaluate your outputs
- **Institutional Memory**: the implicit heuristics an organization develops by shipping many iterations and seeing what breaks in the real world
- **Systems Architecture**: the design that allows each of these levers is used to its fullest potential. (there is often much, much, MUCH more that you can get out of your data than you might realize)

Together, these form what I call **Judgment Capital**—a compound, reusable asset that lets you consistently measure and enforce quality in ways your competitors just can't copy.

This grows slowly, resists leakage, and amortizes over every feature—actively used evaluators compound.

## Measurement as Bottleneck

Any complex pipeline is bottlenecked by its hardest atomic subtask.

For an AI pipeline in a less-easy-to-verify-domain, that subtask is often _evaluating the output_.

For example—say you created a validator with 80% accuracy with a simple "correctness" metric. This validator limits you to shipping only what you can prove meets that bar. There is room for LLM variance creating incredible results using your pipeline, but you will only stumble across "incredible"—never reliably distinguish, reproduce, or guarantee at scale. - > What you choose to measure—and **how reliably** you measure it—directly caps how good your product can become.

If all you check is correctness, or if you produce "good" or "good enough" outputs, you can never tell if you're producing "great", "incredible", or "superhuman" outputs—they all look the same.

## What You Measure, You Optimize

Benchmarks are saturating faster and faster. The fastest way to improve performance on a given domain is to measure it—and you cannot improve on what you cannot measure!

![AI benchmarks showing rapid progress towards human performance across multiple domains](./ai-benchmarks-human-performance.png)

Lately, LLMs have been exhibiting the marvelous property of autonomously being able to make numbers go up and tests go green—from CUDA kernels to tested software generation to DSPy's optimizers. Self improvement is possible when you constrain the domain well enough.

Reliable verification unlocks your ability to define, automate, and scale excellence. It allows your models to autonomously optimize—achieving outcomes you didn't even realize were possible.

Both LLMs and LLM engineers are great at hillclimbing. By generating quality data and evaluating it properly, it becomes an OOM more straightforward to reach the Pareto Frontier at the intersection of the metrics you and your customers actually care about.

## RL & Optimization

What do you get when you measure performance?

At the lowest-effort end of the spectrum, **it lets you consistently tell if a given output is good**, because you now have things you're looking for, and when there's a corner case, you can reason about it easier and document it for the future.

With a little work, a calibrated scorer, and representative inputs, it lets you do things like **optimizing prompt pipelines to excel on any chosen metric** (accuracy, latency, you name it).

With a lot of work and a directionally correct scorer, it **gives your model powerful reward signal** when what you're looking for is so nuanced that even you cannot tell yourself. ex: in GRPO, the relative advantage in a group matter most—so **even if you're not perfect**, if instead of you, an LLM can judge, you can provide useful RL signal.

Both LLM pipelines and RL training pipelines are (or can be) trained to optimize towards clearly defined objectives. **The evaluators of these objectives are the what shapes the outcomes.**

Say we wanted "superhuman" outcomes—these are very difficult to hand-design, because by definition, they exceed human capability.

However, we **can** define, directionally, aspects of what that looks like—or what separates worse outputs from better ones! This puts us squarely on the path towards superhuman on your domain :)

## A Brief Aside

This is _a_ path to building **the best** product in your field.

Remember the last time you were surprised and awed by this technology? This is the most magical stuff we've ever built. And roughly every 6 months, even as we get used to the pace of progress, you have a "holy fuck this shit is incredible" moment. You've experienced the ones I'm talking about!

It's possible to reliably replicate that feeling for your customers. You can create an unimaginably, fucking incredible thing, whatever it is you want to make—one that really takes advantage of **superhuman** intelligence on tap.

Dude, the thing is, nobody knows what that will look like. Do you remember the best food you've ever had? Okay, so for me, they're mulitas, from this place called Tacos Maggie. And they're just so, so, so fucking incredibly good. Like you have not lived until you've tried their Tripas Mulitas. I search for better ones everywhere I go and all I find is dissapointment. Heaven on earth, this truck is an absolute pillar of the community.

Much in the same way—it is very, very possible to create incredible experiences with the things it is you're building. Like what does the best doctor's visit you've ever had look like? What does  superhuman looks like at the things that your company is an expert on? Wouldn't you want to know?

## "Good Enough" Isn't Enough

If you, by hand, can reliably tell what's good enough, then why invest in any infra or thinking about this problem at all?

As soon as you scale or aim higher, the uncertainty about the performance of your pipeline makes improving it impossible. It makes incredible results for your users a rare, sporadic occurence, as opposed to engineering reliable moments of surprise and delight.

And it really, really is trivial to get started. Pick the things you vibe-test on. Make it formal. Write up a prompt that gets you >=50%. You then have something to improve, which is much, much, much better than nothing, and the first path towards **reliable**, incredible AI products that are incredibly hard to replicate.

## The Highest-Leverage Move

The AI companies that create incredible products will be those who reliably define, measure, and optimize towards excellence.

I hope this is obvious. The best teams do this already. They've always done this. It's just a little harder now, but is just as important a philosophy to hold.

AI engineering is… normal engineering, just with less reliable systems we're working with. But making them reliable is an engineering challenge that unlocks a lot of the benefits of AI that we've been promised and have had hyped up but haven't seen.

This is a tricky, hard problem that is incredibly rewarding to make progress on and do right. Starting is deceptively easy. And it's really simple to start :)

The most defensible moat in AI is knowing how to measure and improve what matters most—because the path to superhuman outcomes always runs straight through superhuman evaluation.
