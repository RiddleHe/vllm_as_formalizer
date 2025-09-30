(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        green - block
        orange - block
        red - block
        yellow - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable green)
        (on orange green)
        (on red orange)
        (on yellow red)
        (on blue yellow)
        (clear blue)
        ; purple
        ; green -> orange -> red -> yellow -> blue
    )
    (:goal (and (on red blue)
        (on orange purple)
        (on green orange)
        (on yellow green)))
)
    