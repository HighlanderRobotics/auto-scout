// Reads the REBUILT broadcast score overlay from cropped frames with Apple Vision.
// Usage: swift ocr_overlay.swift overlay/   -> CSV on stdout
import Foundation
import Vision
import AppKit

let dir = CommandLine.arguments[1]
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
print("video_s,blue_score,clock,red_score,shift,shift_clock,raw")
for (i, f) in files.enumerated() {
    let url = URL(fileURLWithPath: dir).appendingPathComponent(f)
    guard let img = NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    try VNImageRequestHandler(cgImage: img).perform([req])
    // Crop is 1160 wide, starting at x=380 in the 1920 frame. Bounding boxes are normalized, origin bottom-left.
    var blue = "", clock = "", red = "", shift = "", shiftClock = ""
    var raw: [String] = []
    for obs in req.results ?? [] {
        guard let t = obs.topCandidates(1).first?.string else { continue }
        let b = obs.boundingBox
        let x = 380 + b.midX * 1160, top = b.midY > 0.45
        raw.append("\(t)@\(Int(x))\(top ? "T" : "B")")
        if top {
            if x > 780 && x < 900 { blue = t }
            else if x >= 900 && x <= 1020 && t.contains(":") { clock = t }
            else if x > 1020 && x < 1140 { red = t }
        } else if x >= 890 && x <= 1030 {
            if t.contains("/") {
                let parts = t.components(separatedBy: CharacterSet(charactersIn: " :")).filter { !$0.isEmpty }
                shift = parts.first(where: { $0.contains("/") }) ?? t
                if t.contains(":") { shiftClock = String(t.split(separator: ":").last ?? "") }
                if shift == "/" || !shift.contains(where: \.isNumber) { shift = t.replacingOccurrences(of: " ", with: "").components(separatedBy: ":").first ?? t }
            } else if t.contains(":") { shiftClock = t.replacingOccurrences(of: ":", with: "") }
        }
    }
    print("\(i),\(blue),\(clock),\(red),\(shift),\(shiftClock),\"\(raw.joined(separator: " | "))\"")
}
