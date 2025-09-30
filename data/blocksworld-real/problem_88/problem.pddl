(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        blue - block
        orange - block
        purple - block
        yellow - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on blue green)
        (on orange blue)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable yellow)
        (on red yellow)
        (clear red)
        ; green -> blue -> orange
        ; purple
        ; yellow -> red
    )
    (:goal (and (on green blue)
        (on purple orange)
        (on red purple)
        (on yellow red)))
)
    