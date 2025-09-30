(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        red - block
        green - block
        purple - block
        blue - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on red yellow)
        (on green red)
        (on purple green)
        (clear purple)
        (ontable blue)
        (on orange blue)
        (clear orange)
        ; yellow -> red -> green -> purple
        ; blue -> orange
    )
    (:goal (and (on orange purple)
        (on green blue)
        (on yellow red)))
)
    