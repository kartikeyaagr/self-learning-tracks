---
id: "11.09"
course: "system-design-101"
module: "11"
slug: "photo-tagging-coordinates"
title: "Photo tagging coordinates"
keyIdeas: ["Photo tags mark a person or item at a point or box on an image.", "Store coordinates as fractions of the image size (x and y between 0 and 1), not pixels, because the same image is shown at many sizes and resolutions.", "Data model: tag(photo_id, tagged_user_id, x, y, created_by, status), with status for approval if the tagged person must confirm.", "Index by tagged_user_id to show \"photos of me\".", "Validate coordinates and handle images that are cropped or rotated after tagging."]
question: "Why store tag positions as fractions (0-1) of image size instead of pixels?"
answer: "The same photo is displayed at many sizes (thumbnails, phone, desktop, retina, the original, cropped previews), so a pixel position is only correct for the exact size it was captured at, and lands in the wrong place or off the image everywhere else; it is even ambiguous between CSS and device pixels. Fractions of width and height are independent of resolution: one stored value maps to any rendition by multiplying by its displayed width and height, including renditions and formats created later, and invalid values are easy to detect because they fall outside 0 to 1. Crops, rotation and letterboxing change the frame, so you map through the crop rectangle, normalise EXIF orientation on upload and use the image's actual rendered rectangle."
sources: [{"title": "MDN: Coordinate systems", "url": "https://developer.mozilla.org/en-US/docs/Web/CSS/CSSOM_view/Coordinate_systems", "note": "Why to store relative coordinates."}]
tags: ["Design"]
---

## The problem

Photo tagging stores "Alice is at this point (or in this box) in photo 123". It looks like the simplest data model in this module: a photo ID, a user ID and a position. The catch is in the word "position". The same photo is shown at many sizes:

- a 4000 × 3000 original from the camera,
- a 1080-pixel-wide version on phones,
- a 320-pixel thumbnail in the feed,
- a 2× version for high-density ("retina") screens,
- a cropped square preview in a profile grid,
- and future renditions nobody has designed yet (a new tablet layout, a new image format).

The tag marker must land on Alice's face in all of them, and must keep doing so when the image pipeline changes. The answer is to store positions as **fractions** of the image, not pixels. This lesson explains why, and covers the cases where fractions alone aren't enough: crops, rotation and letterboxing.

## Coordinates are relative to an origin

A position on the web is an offset from an **origin**, by default the **top-left corner**, with x increasing to the right and y **downward** (unlike school graphs, where y goes up). The same click has different coordinates depending on which origin you measure from. A mouse event in the browser carries several:

~~~dg-layers The same click measured from four different origins. Only coordinates relative to the image itself mean anything for a tag.
screenX / screenY | from the physical screen's corner
pageX / pageY | from the top of the whole document
clientX / clientY | from the viewport; changes as you scroll
*offsetX / offsetY | from the target element's padding edge
~~~

For tagging, you want coordinates relative to the **rendered image itself**. The robust way is to use the image's bounding box: `rect = img.getBoundingClientRect()`, then `x = (event.clientX − rect.left) / rect.width` and `y = (event.clientY − rect.top) / rect.height`. That gives fractions directly and works even when the image is scaled by CSS or transformed, where `offsetX` can be surprising (it measures from the padding edge, so borders and padding shift it).

## Pixels break across sizes

Suppose a user clicks Alice's face at (540, 300) on an image displayed 1080 × 810 pixels. If you store (540, 300):

- On the 320 × 240 thumbnail, (540, 300) is **off the image** entirely.
- On the 4000 × 3000 original, it is in the top-left area, nowhere near Alice.
- On a high-density screen, CSS pixels and device pixels differ (a 2× screen has four device pixels per CSS pixel), so even "the same" display size can disagree on raw pixel numbers depending on which kind you recorded.
- If the thumbnail pipeline is later changed from 320 to 360 pixels wide, every stored tag is wrong.

You could store pixels plus the size they were measured at, and rescale on every read. That is the fraction approach with extra steps and more ways to get it wrong.

## Fractions are size-independent

Store the position as a **fraction** of the image's width and height, each between 0 and 1. These are often called **normalised coordinates**:

`x = 540 / 1080 = 0.5`, `y = 300 / 810 ≈ 0.370`

To draw the tag on any rendition, multiply by that rendition's displayed size. On the 320 × 240 thumbnail the marker goes at (160, 89); on the 4000 × 3000 original at (2000, 1111). A box tag stores (x, y, width, height), all as fractions.

~~~dg-flow One tag, many sizes: capture fractions once, then multiply by whatever size each rendition is displayed at.
[Capture] Click at 540, 300 on 1080 x 810 -(divide by size)-> *Stored: x 0.50, y 0.37
[Thumbnail] Stored fractions -(times 320 x 240)-> Marker at 160, 89
[Original] Stored fractions -(times 4000 x 3000)-> Marker at 2000, 1111
~~~

Why this is the right representation:

- **Resolution independence.** Any rendition with the same aspect ratio maps correctly, including ones created later.
- **Responsive layouts.** The client only needs the rendered size, which it always knows, not the original's dimensions.
- **Re-encoding and resizing.** Regenerating thumbnails, serving WebP or AVIF instead of JPEG, or changing quality settings changes nothing.
- **Simple validation.** Anything outside 0 to 1 is invalid, and a box must satisfy x + width ≤ 1 and y + height ≤ 1.
- **Portable.** Mobile apps, web clients and server-side renderers (say, generating a share image with name labels) all use the same numbers.

~~~dg-compare Storing a tag as pixels versus as fractions of the image.
[Pixels]
540, 300 measured on a 1080 x 810 display
- Wrong on every other size
- Ambiguous: CSS or device pixels?
- Breaks when renditions change
+ Matches what the click handler reports
[*Fractions]
0.50, 0.37 of width and height
+ Same on every rendition
+ Survives new sizes and formats
+ Easy to validate, 0 to 1
- Crops and rotation need explicit mapping
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a tag stored and drawn**

Stored row: `{photo_id: 123, user_id: alice, x: 0.5000, y: 0.3704, w: 0.0800, h: 0.1100}`.

Feed card renders the photo at 375 × 281: box at (0.5 × 375, 0.3704 × 281) = (187.5, 104.1), size (0.08 × 375, 0.11 × 281) = 30 × 31 pixels.

Desktop lightbox at 1600 × 1200: box at (800, 444.5), size 128 × 132.

One row serves every client and every future rendition.

</div>

## What fractions don't fix

Fractions are relative to a particular **frame**: the full, upright, uncropped image. Anything that changes the frame needs an explicit mapping.

### Crops

A square feed preview of a landscape photo shows only part of the image. Store the crop rectangle in fractions too, then map a tag into the crop's own frame:

`x_in_crop = (x − crop.x) / crop.width`, `y_in_crop = (y − crop.y) / crop.height`

Tags whose mapped coordinates fall outside 0 to 1 are outside the crop and are hidden.

<div class="ex">
<div class="lab">Worked example</div>

**a centred square crop**

The original is 4000 × 3000. A centred square crop keeps x from 500 to 3500 and all of y. As fractions: crop.x = 0.125, crop.width = 0.75, crop.y = 0, crop.height = 1.

Alice's tag at x = 0.5 maps to (0.5 − 0.125) / 0.75 = 0.5: still centred. Bob's tag at x = 0.1 maps to (0.1 − 0.125) / 0.75 = −0.033: outside the crop, so the square preview hides his marker, and the full view shows it.

</div>

### Rotation and EXIF orientation

Phone cameras often save the pixels sideways and add an **EXIF orientation** flag telling viewers to rotate on display. If one client tags the raw sideways pixels and another the rotated view, their fractions disagree. Normalise the image (apply the rotation and strip the flag) when it is uploaded, before anyone can tag it, so every client sees the same upright frame. If a user later rotates a photo 90° clockwise, transform the stored tags: `x' = 1 − y`, `y' = x` (the top-left corner moves to the top-right), and swap width and height for boxes.

### Aspect ratio and letterboxing

Fractions assume each rendition has the same aspect ratio as the original. If a layout fits a 4:3 photo into a square box with `object-fit: contain`, the image is **letterboxed** (bars above and below), and the box's size is not the image's size. Compute the image's actual rendered rectangle inside the container and map into that.

~~~dg-matrix What each display transformation does to stored fractions, and what to do about it.
Change | Fractions still valid? | What to do
Resize, same aspect ratio | yes | multiply by rendered size
New format or quality | yes | nothing
Crop | partial | map into crop frame, hide outside
Rotate 90 degrees | no | transform: x' = 1 - y, y' = x
EXIF orientation flag | no, if clients disagree | normalise on upload
Letterbox, object-fit contain | partial | map into the image's rendered rectangle
Stretch to a new aspect ratio | partial, distorted | avoid; preserve aspect ratio
~~~

<div class="ex">
<div class="lab">Worked example</div>

**drawing a tag in a letterboxed view**

A 4000 × 3000 photo is shown with `object-fit: contain` in a 400 × 400 box. Scale = min(400 / 4000, 400 / 3000) = 0.1, so the image renders at 400 × 300, centred with 50-pixel bars above and below.

Alice's tag at (0.5, 0.37) is drawn at x = 0 + 0.5 × 400 = 200 and y = 50 + 0.37 × 300 = 161. Using the box's 400-pixel height instead of the image's 300 would put the marker at y = 148, 13 pixels too high, on Alice's forehead.

</div>

## Storage and precision

A tag row needs photo ID, tagged user ID, tagger ID, the position and, for boxes, a size, plus a status (pending approval, approved, removed), since many platforms let the tagged person approve or remove tags. Index by photo (to draw tags on a photo) and by tagged user (for "photos of Alice"), the same two-direction pattern as follows and reactions.

Precision is a non-issue: a 32-bit float has about 7 significant digits, and even a fixed-point integer (fraction × 10,000, stored as a 16-bit or 32-bit integer) is plenty: 0.0001 of a 4000-pixel image is 0.4 pixels. Fixed-point integers compare exactly and avoid float formatting surprises in JSON.

## Common mistakes

- **Storing pixels from whatever size the tagger saw.** Correct only on that one size.
- **Using offsetX without thinking about borders and padding**, or clientX without subtracting the image's position.
- **Ignoring EXIF orientation**, so tags made on iOS appear sideways on web.
- **Using the container's size** instead of the image's rendered rectangle when the image is letterboxed.
- **Forgetting crops**, drawing markers outside a cropped preview or in the wrong place within it.

## In an interview

Say "store normalised coordinates: x, y, width, height as fractions of the upright original, and multiply by the rendered size on each client". Then mention the edge cases unprompted: normalise EXIF orientation on upload, map through crop rectangles, compute the rendered rectangle under letterboxing. It shows you've thought about the whole pipeline, not just the happy path.

<div class="take">
<div class="lab">Key takeaway</div>

browser coordinates are offsets from a top-left origin and depend on the reference element and display size, so pixels are only right for one rendition. Store tag positions (and crop boxes) as fractions 0 to 1 of the upright original's width and height, then multiply by each rendition's rendered size, handling crops, rotation and letterboxing explicitly.

</div>
