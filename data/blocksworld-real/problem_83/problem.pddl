(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        red - block
        yellow - block
        blue - block
        green - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on red purple)
        (on yellow red)
        (on blue yellow)
        (on green blue)
        (clear green)
        (ontable orange)
        (clear orange)
        ; purple -> red -> yellow -> blue -> green
        ; orange
    )
    (:goal (and (on yellow green)
        (on orange yellow)
        (on blue orange)
        (on red blue)
        (on purple red)))
)
    